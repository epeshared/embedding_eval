# AMX 版
mkdir -p llama-libs

[ -d llama.cpp/.git ] || git clone https://github.com/ggml-org/llama.cpp 
cd llama.cpp
rm -rf build-amx
rm -rf build-noamx 

cmake -B build-amx -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON -DGGML_AVX512=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-amx -j
cp build-amx/bin/libllama.so ../llama-libs/libllama_amx.so

# NO-AMX 版
cmake -B build-noamx -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-noamx -j
cp build-noamx/bin/libllama.so ../llama-libs/libllama_noamx.so
