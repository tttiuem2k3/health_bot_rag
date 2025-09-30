import pkg_resources

# Danh sách thư viện cần kiểm tra với phiên bản yêu cầu
required_packages = {
    "torch": "2.2.2",
    "transformers": "4.39.3",
    "accelerate": "0.28.0",
    "bitsandbytes": "0.43.0",
    "huggingface-hub": "0.22.2",
    "langchain": "0.1.14",
    "langchain-core": "0.1.43",
    "langchain-community": "0.0.31",
    "pypdf": "4.2.0",
    "sentence-transformers": "2.6.1",
    "beautifulsoup4": "4.12.3",
    "langserve": None,  
    "chromadb": "0.4.24",
    "langchain-chroma": "0.1.0",
    "faiss-cpu": "1.8.0",
    "rapidocr-onnxruntime": "1.3.16",
    "unstructured": "0.13.2",
    "fastapi": "0.110.1",
    "uvicorn": "0.29.0"
}

for package, required_version in required_packages.items():
    try:
        dist = pkg_resources.get_distribution(package)
        installed_version = dist.version
        if required_version is None:
            print(f"{package}: Đã cài phiên bản {installed_version}")
        elif installed_version == required_version:
            print(f"{package}: Đúng phiên bản ({installed_version})")
        else:
            print(f"{package}: Sai phiên bản ({installed_version}) - yêu cầu {required_version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Chưa tồn tại")
