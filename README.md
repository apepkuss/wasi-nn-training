# wasmedge-nn-training

> This is an experimental project, and it is still in the active development. 

The goal of this project is to explore the feasibility of providing AI training capability based on WasmEdge Plugin mechanism. This project consists of two major projects: 

- `tch-backend-plugin` constructs a plugin prototype integrated with PyTorch.

- `test-tch-backend` is a wasm app that is responsible for preparing data and calling the `train` interface to trigger a training task.

## Requirements

- OS: Ubuntu 20.04+

- Mnist image data

    The Mnist image data is located in the `data` directory of this repo.

- Install libtorch

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
wasmedge --dir .:. target/wasm32-wasi/release/test-tch-backend.wasm
```

### Result

```bash
[Wasm] Preparing training images ... [Done]
[Wasm] Preparing training labels ... [Done]
[Wasm] Preparing test images ... [Done]
[Wasm] Preparing test lables ... [Done]

*** Welcome! This is `wasmedge-nn-training` plugin. ***

[Plugin] Preparing train images ... [Done] (shape: [60000, 1, 28, 28], dtype: Float)
[Plugin] Preparing train labels ... [Done] (shape: [60000], dtype: Int64)
[Plugin] Preparing test images ... [Done] (shape: [10000, 1, 28, 28], dtype: Float)
[Plugin] Preparing test labels ... [Done] (shape: [10000], dtype: Int64)
[Plugin] Labels: 10
[Plugin] Device: Cpu
[Plugin] Learning rate: 0.0001
[Plugin] Epochs: 10
[Plugin] batch size: 128
[Plugin] Initial accuracy:  9.27%
[Plugin] Start training ... 
        epoch:    1 test acc: 87.15%
        epoch:    2 test acc: 89.22%
        epoch:    3 test acc: 89.97%
        epoch:    4 test acc: 90.79%
        epoch:    5 test acc: 91.12%
        epoch:    6 test acc: 91.09%
        epoch:    7 test acc: 91.22%
        epoch:    8 test acc: 91.08%
        epoch:    9 test acc: 91.23%
[Plugin] Finished
[Plugin] The pre-trained model is dumped to `trained_model.pt`
```


