function Loader() {
    let self = this;
    let asmPath = "";
    let wasmPath = "";
    let simdPath = "";
    let threadsPath = "";
    let mtSIMDPath = "";
    
    
    this.judgeWASM = function() {
        try{
            let test = WebAssembly;
            return true;
        } catch(e) {
            return false;
        }
    }

    this.judgeSIMD = function() {
        return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]));
    }

    this.judgeThreads = function() {
        try {
            let test1 = (new MessageChannel).port1.postMessage(new SharedArrayBuffer(1));
            let result = WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]));
            return result;
        } catch(e) {
            return !1
        }
    }

    this.setPaths = function(paths) {
        if(!(paths instanceof Object)) {
            throw new Error("The input should be a object that points the path to the OpenCV.js");
        }

        if ("asm" in paths) {
            self.asmPath = paths["asm"];
        }

        if ("wasm" in paths) {
            self.wasmPath = paths["wasm"];
        }

        if ("threads" in paths) {
            self.threadsPath = paths["threads"];
        }

        if ("simd" in paths) {
            self.simdPath = paths["simd"];
        }

        if ("mtSIMD" in paths) {
            self.mtSIMDPath = paths["mtSIMD"];
        }
    }

    this.loadOpenCV = function (onloadCallback) {
        let OPENCV_URL = "";
        let wasmSupported = self.judgeWASM();
        let simdSupported = self.judgeSIMD();
        let threadsSupported = self.judgeThreads();

        if (!wasmSupported && OPENCV_URL == "" && self.asmPath != "") {
            OPENCV_URL = asmPath;
        } else if (!wasmSupported && self.asmPath == ""){
            throw new Error("The browser supports the Asm.js only, but the path of OpenCV.js for Asm.js is empty");
        }

        if (simdSupported && threadsSupported && OPENCV_URL == "" && self.mtSIMDPath != "") {
            OPENCV_URL = self.mtSIMDPath;
        } else if (simdSupported && threadsSupported && OPENCV_URL == "") {
            throw new Error("The browser supports simd and threads, but the path of OpenCV.js with simd and threads optimization is empty");
        }

        if (simdSupported && OPENCV_URL == "" && self.simdPath != "") {
            OPENCV_URL = self.simdPath;
        } else if (simdSupported && OPENCV_URL == "") {
            throw new Error("The browser supports simd, but the path of OpenCV.js with simd optimization is empty");
        }

        if (threadsSupported && OPENCV_URL == "" && self.threadsPath != "") {
            OPENCV_URL = self.threadsPath;
        } else if (threadsSupported && OPENCV_URL == "") {
            throw new Error("The browser supports threads, but the path of OpenCV.js with threads optimization is empty");
        }

        if (wasmSupported && OPENCV_URL == "" && self.wasmPath != "") {
            OPENCV_URL = self.wasmPath;
        } else if (wasmSupported && OPENCV_URL == "") {
            throw new Error("The browser supports wasm, but the path of OpenCV.js with wasm is empty");
        }

        let script = document.createElement('script');
        script.setAttribute('async', '');
        script.setAttribute('type', 'text/javascript');
        script.addEventListener('load', () => {
            onloadCallback();
        });
        script.addEventListener('error', () => {
            console.log('Failed to load opencv.js');
        });
        script.src = OPENCV_URL;
        let node = document.getElementsByTagName('script')[0];
        node.parentNode.insertBefore(script, node);
    }
}