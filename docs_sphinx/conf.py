DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,imgproc,dnn").split(",")
    if m.strip()
] 
