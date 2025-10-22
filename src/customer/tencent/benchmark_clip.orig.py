import os
import numpy as np
import time
import transformers
import asyncio
import concurrent.futures
import argparse
import torch

try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    IPEX_AVAILABLE = False


def prepare_huggingface_model(
    pretrained_model_name_or_path: str | os.PathLike, **model_params
):
    """
    Prepare huggingface model and processor.
    """
    processor = transformers.AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )

    config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )
    if hasattr(config, "auto_map"):
        class_name = next(
            (k for k in config.auto_map if k.startswith("AutoModel")), "AutoModel"
        )
    else:
        # TODO: What happens if more than one
        class_name = config.architectures[0]
    model_class = getattr(transformers, class_name)
    model = model_class.from_pretrained(pretrained_model_name_or_path, **model_params)

    return (model, processor)


class BenchmarkImage:
    def __init__(self, device: str, data_type: str):
        self.text = "a beautiful landscape"
        self.model, self.processor = prepare_huggingface_model(
            "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",
            device_map=device,
        )
        print(f"IPEX_AVAILABLE=", IPEX_AVAILABLE)
        if data_type == "bf16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16, inplace=True)
                self.model.to(torch.bfloat16)
            else:
                self.model.to(torch.bfloat16)
        elif data_type == "fp16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(self.model, dtype=torch.float16, inplace=True)
                self.model.to(torch.float16)
            else:
                self.model.to(torch.float16)

        # self.model = torch.compile(self.model)

        print("Init Finished!")

    def benchmark(self, inputs):
        inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        _ = outputs.logits_per_text


async def run_benchmark_async(instance: BenchmarkImage, batch_size, num_iter, executor):
    loop = asyncio.get_event_loop()

    def run_benchmark():
        images = [
            np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
            for _ in range(batch_size)
        ]
        inputs = instance.processor(
            text=[instance.text], images=images, return_tensors="pt", padding=True
        )

        for i in range(num_iter):
            instance.benchmark(inputs)

    result = await loop.run_in_executor(executor, run_benchmark)
    return result


async def benchmark_image_async(parallelism, batch_size, num_iter, device, data_type):
    print(
        f"开始异步并行测试: parallelism={parallelism}, batch_size={batch_size}, num_iter={num_iter}, device={device}, data_type={data_type}"
    )
    benchmark_image = [None] * parallelism
    for i in range(parallelism):
        benchmark_image[i] = BenchmarkImage(device, data_type)

    # 创建线程池执行器
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        tasks = []
        for i in range(parallelism):
            task = run_benchmark_async(
                benchmark_image[i], batch_size, num_iter, executor
            )
            tasks.append(task)

        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        print(f"total time: {total_time}")


def benchmark_image(parallelism, batch_size, num_iter, device, data_type):
    asyncio.run(
        benchmark_image_async(parallelism, batch_size, num_iter, device, data_type)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP model with random data")
    parser.add_argument(
        "--parallelism", type=int, default=1, help="Parallelism of the benchmark"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size of the model"
    )
    parser.add_argument(
        "--num_iter", type=int, default=5, help="Number of iterations of the model"
    )
    parser.add_argument("--device", type=str, default="auto", help="Target device")
    parser.add_argument("--data_type", type=str, default="fp32")

    args = parser.parse_args()

    benchmark_image(
        parallelism=args.parallelism,
        batch_size=args.batch_size,
        num_iter=args.num_iter,
        device=args.device,
        data_type=args.data_type,
    )
