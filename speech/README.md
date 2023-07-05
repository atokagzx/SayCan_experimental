Running:

```bash
pip3 install grpcio-tools
python3 -m grpc_tools.protoc -I ./grpc --python_out=. --grpc_python_out=. ./grpc/recognition.proto ./grpc/{storage,task}.proto
```