# wasi-nn-training

## Install libtorch

Reference [Libtorch Manual Install](https://github.com/LaurentMazare/tch-rs#libtorch-manual-install)

## Draft a resnet model

The following Python script defines a Resnet model. The last line of the code dumps the model for training.

```python
import torch
from torch.nn import Module


class DemoModule(Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(16 * 28 * 28, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)


traced_script_module = torch.jit.script(DemoModule())
traced_script_module.save("model.pt")
```

## Build the training plugin

```bash
cargo build -p tch-backend-plugin --release

// copy the plugin library into the wasmedge plugin directory
cp ./target/release/libtch_backend_plugin.so /usr/local/lib/wasmedge
```

## Define a wasm app

The wasm app is responsible for loading images from a specified location, preprocessing the image data, and splitting the data into three parts for training, testing and validation. Finally, the wasm app calls the `train` interface exported by the external module *plugin* which is powered by WasmEdge wasi-nn-training plugin.

### Build

```bash
cargo build -p test-tch-backend --target wasm32-wasi --release
```

## Train

```bash
wasmedge --dir .:. ./target/wasm32-wasi/release/test-tch-backend.wasm
```

### Result

```bash
[Wasm] Preparing training images ... [Done] (shape: [60000, 1, 28, 28], dtype: f32)
[Wasm] Preparing training images ... [Done] (shape: [60000], dtype: i64)
[Wasm] Preparing training images ... [Done] (shape: [10000, 1, 28, 28], dtype: f32)
[Wasm] Preparing training images ... [Done] (shape: [10000], dtype: i64) 

*** Welcome! This is `train` host function in `wasi-nn-training` plugin. ***

[Plugin] Converting training image data to tch::Tensor ... [Done] (shape: [60000, 1, 28, 28], dtype: Float) 
[Plugin] Converting training label data to tch::Tensor ... [Done] (shape: [60000], dtype: Int64) 
[Plugin] Converting test image data to tch::Tensor ... [Done] (shape: [10000, 1, 28, 28], dtype: Float) 
[Plugin] Converting test label data to tch::Tensor ... [Done] (shape: [10000], dtype: Int64) 
[Plugin] Labels: 10
[Plugin] Device: Cpu
[Plugin] Initial accuracy:  9.47%
[Plugin] Start training ... 
        epoch:    1 test acc: 86.83%
        epoch:    2 test acc: 89.67%
        epoch:    3 test acc: 90.36%
        epoch:    4 test acc: 89.90%
        epoch:    5 test acc: 90.88%
        epoch:    6 test acc: 91.12%
        epoch:    7 test acc: 91.11%
        epoch:    8 test acc: 91.18%
        epoch:    9 test acc: 91.16%
[Plugin] Finished
[Plugin] The pre-trained model is dumped to `trained_model.pt`
```


