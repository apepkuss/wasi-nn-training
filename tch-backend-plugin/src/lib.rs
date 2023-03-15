use anyhow::Result;
use wasmedge_sdk::{
    error::HostFuncError,
    host_function,
    plugin::{ffi, PluginDescriptor, PluginVersion},
    Caller, ImportObjectBuilder, ValType, WasmValue,
};

use tch::nn::{Adam, ModuleT, OptimizerConfig, VarStore};
use tch::vision::dataset::Dataset;
use tch::TrainableCModule;
use tch::{Device, Tensor};

// A native function to be wrapped as a host function
#[host_function]
fn real_add(_: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("Welcome! This is NaiveMath plugin.");

    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    if input.len() != 2 {
        return Err(HostFuncError::User(1));
    }

    let a = if input[0].ty() == ValType::I32 {
        input[0].to_i32()
    } else {
        return Err(HostFuncError::User(2));
    };

    let b = if input[1].ty() == ValType::I32 {
        input[1].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };

    let c = a + b;
    Ok(vec![WasmValue::from_i32(c)])
}

#[host_function]
fn set_input_tensor(
    caller: Caller,
    input: Vec<WasmValue>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("This is `set_input_tensor` host function.");

    let memory = caller.memory(0).expect("failed to get memory at idex 0");

    if input.len() != 5 {
        return Err(HostFuncError::User(1));
    }

    let dt_offset = if input[0].ty() == ValType::I32 {
        input[0].to_i32()
    } else {
        return Err(HostFuncError::User(2));
    };
    println!("dt_offset: {dt_offset}");

    let dt_size = if input[1].ty() == ValType::I32 {
        input[1].to_i32()
    } else {
        return Err(HostFuncError::User(2));
    };
    println!("dt_size: {dt_size}");

    let dm_offset = if input[2].ty() == ValType::I32 {
        input[2].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };
    println!("dm_offset: {dm_offset}");

    let dm_size = if input[3].ty() == ValType::I32 {
        input[3].to_i32()
    } else {
        return Err(HostFuncError::User(4));
    };
    println!("dm_size: {dm_size}");

    let ty = if input[4].ty() == ValType::I32 {
        input[4].to_i32()
    } else {
        return Err(HostFuncError::User(5));
    };
    println!("ty: {ty}");

    // ======

    let data = memory
        .data_pointer(dt_offset as u32, dt_size as u32)
        .expect("failed to get data from linear memory");

    let slice = unsafe { std::slice::from_raw_parts(data, 8) };
    println!("slice1: {:?}", slice);

    // extract the first (offset, size) from the linear memory
    let offset1 = i32::from_le_bytes(slice[0..4].try_into().unwrap());
    let size1 = i32::from_le_bytes(slice[4..8].try_into().unwrap());
    println!("offset1: {}, size1: {}", offset1, size1);
    // extract the first sequence of numbers from the linear memory by (offset, size)
    let num1 = memory
        .read(offset1 as u32, size1 as u32)
        .expect("failed to read numbers");
    println!("num1: {:?}", num1);

    let dims = memory
        .data_pointer(dm_offset as u32, dm_size as u32)
        .expect("failed to get dims from linear memory");
    let dims = unsafe { std::slice::from_raw_parts(dims, 2) };
    let dims: Vec<i64> = dims.iter().map(|&x| x as i64).collect();
    println!("dims: {:?}", dims);

    let tensor = Tensor::of_slice(num1.as_slice());
    let tensor = tensor.reshape(dims.as_slice());
    println!("tensor: {:?}", tensor);

    Ok(vec![])
}

#[host_function]
fn train(caller: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("This is `set_input_tensor` host function.");

    let memory = caller.memory(0).expect("failed to get memory at idex 0");

    assert_eq!(input.len(), 10);

    // extract train_images
    let train_images: Tensor = {
        let offset = if input[0].ty() == ValType::I32 {
            input[0].to_i32()
        } else {
            return Err(HostFuncError::User(1));
        };
        println!("offset: {offset}");

        let len = if input[1].ty() == ValType::I32 {
            input[1].to_i32()
        } else {
            return Err(HostFuncError::User(2));
        };
        println!("len: {len}");

        // parse train_images

        let data_ptr = memory
            .data_pointer(offset as u32, len as u32)
            .expect("plugin: train_images: failed to get the point to the data");
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, len as usize) };

        let offset_data = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let size_data = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        let data = memory
            .read(offset_data as u32, size_data as u32)
            .expect("plugin: train_images: failed to extract tensor data");
        println!(
            "plugin: train_images: data: {:?}, len: {}",
            data,
            data.len()
        );

        // extract tensor's dimensions
        let offset_dims = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let size_dims = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        let dims = memory
            .read(offset_dims as u32, size_dims as u32)
            .expect("plugin: train_images: faied to extract tensor dimensions");
        let dims: Vec<i64> = protocol::bytes_to_i32_vec(dims.as_slice())
            .iter()
            .map(|&c| c as i64)
            .collect();
        println!("plugin: train_images: dims: {:?}", dims);

        // extract tensor's dtype
        let dtype = slice[16];
        println!("plugin: train_images: dtype: {dtype}");

        to_tch_tensor(dtype, dims.as_slice(), data.as_slice())
    };

    // extract train_labels
    let train_labels: Tensor = {
        let offset = if input[2].ty() == ValType::I32 {
            input[2].to_i32()
        } else {
            return Err(HostFuncError::User(3));
        };
        println!("offset: {offset}");

        let len = if input[3].ty() == ValType::I32 {
            input[3].to_i32()
        } else {
            return Err(HostFuncError::User(4));
        };
        println!("len: {len}");

        // parse train_labels

        let data_ptr = memory
            .data_pointer(offset as u32, len as u32)
            .expect("plugin: train_labels: failed to get the point to the data");
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, len as usize) };

        let offset_data = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let size_data = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        let data = memory
            .read(offset_data as u32, size_data as u32)
            .expect("plugin: train_labels: failed to extract tensor data");
        println!(
            "plugin: train_labels: data: {:?}, len: {}",
            data,
            data.len()
        );

        // extract tensor's dimensions
        let offset_dims = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let size_dims = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        let dims = memory
            .read(offset_dims as u32, size_dims as u32)
            .expect("plugin: train_labels: faied to extract tensor dimensions");
        let dims: Vec<i64> = protocol::bytes_to_i32_vec(dims.as_slice())
            .iter()
            .map(|&c| c as i64)
            .collect();
        println!("plugin: train_labels: dims: {:?}", dims);

        // extract tensor's dtype
        let dtype = slice[16];
        println!("plugin: train_labels: dtype: {dtype}");

        to_tch_tensor(dtype, dims.as_slice(), data.as_slice())
    };

    // extract test_images
    let test_images: Tensor = {
        let offset = if input[4].ty() == ValType::I32 {
            input[4].to_i32()
        } else {
            return Err(HostFuncError::User(5));
        };
        println!("offset: {offset}");

        let len = if input[5].ty() == ValType::I32 {
            input[5].to_i32()
        } else {
            return Err(HostFuncError::User(6));
        };
        println!("len: {len}");

        // parse test_images

        let data_ptr = memory
            .data_pointer(offset as u32, len as u32)
            .expect("plugin: test_images: failed to get the point to the data");
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, len as usize) };

        let offset_data = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let size_data = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        let data = memory
            .read(offset_data as u32, size_data as u32)
            .expect("plugin: test_images: failed to extract tensor data");
        println!("plugin: test_images: data: {:?}, len: {}", data, data.len());

        // extract tensor's dimensions
        let offset_dims = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let size_dims = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        let dims = memory
            .read(offset_dims as u32, size_dims as u32)
            .expect("plugin: test_images: faied to extract tensor dimensions");
        let dims: Vec<i64> = protocol::bytes_to_i32_vec(dims.as_slice())
            .iter()
            .map(|&c| c as i64)
            .collect();
        println!("plugin: test_images: dims: {:?}", dims);

        // extract tensor's dtype
        let dtype = slice[16];
        println!("plugin: test_images: dtype: {dtype}");

        to_tch_tensor(dtype, dims.as_slice(), data.as_slice())
    };

    // extract test_labels
    let test_labels: Tensor = {
        let offset = if input[6].ty() == ValType::I32 {
            input[6].to_i32()
        } else {
            return Err(HostFuncError::User(7));
        };
        println!("offset: {offset}");

        let len = if input[7].ty() == ValType::I32 {
            input[7].to_i32()
        } else {
            return Err(HostFuncError::User(8));
        };
        println!("len: {len}");

        // parse test_labels

        let data_ptr = memory
            .data_pointer(offset as u32, len as u32)
            .expect("plugin: test_labels: failed to get the point to the data");
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, len as usize) };

        let offset_data = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let size_data = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        let data = memory
            .read(offset_data as u32, size_data as u32)
            .expect("plugin: test_labels: failed to extract tensor data");
        println!("plugin: test_labels: data: {:?}, len: {}", data, data.len());

        // extract tensor's dimensions
        let offset_dims = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let size_dims = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        let dims = memory
            .read(offset_dims as u32, size_dims as u32)
            .expect("plugin: test_labels: faied to extract tensor dimensions");
        let dims: Vec<i64> = protocol::bytes_to_i32_vec(dims.as_slice())
            .iter()
            .map(|&c| c as i64)
            .collect();
        println!("plugin: test_labels: dims: {:?}", dims);

        // extract tensor's dtype
        let dtype = slice[16];
        println!("plugin: test_labels: dtype: {dtype}");

        to_tch_tensor(dtype, dims.as_slice(), data.as_slice())
    };

    let labels = if input[8].ty() == ValType::I64 {
        input[8].to_i64()
    } else {
        return Err(HostFuncError::User(9));
    };
    println!("labels: {labels}");

    let ds = Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels,
    };

    let device_id = if input[9].ty() == ValType::I32 {
        input[9].to_i32()
    } else {
        return Err(HostFuncError::User(10));
    };
    let device = match device_id {
        0 => Device::Cpu,
        _ => panic!("unsupported device id: {device_id}"),
    };
    println!("device: {:?}", device);

    train_model(ds, device).expect("failed to train model");

    Ok(vec![])
}

/// Defines Plugin module instance
unsafe extern "C" fn create_test_module(
    _arg1: *const ffi::WasmEdge_ModuleDescriptor,
) -> *mut ffi::WasmEdge_ModuleInstanceContext {
    let module_name = "naive-math";
    let import = ImportObjectBuilder::new()
        // add a function
        .with_func::<(i32, i32), i32>("add", real_add)
        .expect("failed to create host function: add")
        .with_func::<(i32, i32, i32, i32, i32), ()>("set_input_tensor", set_input_tensor)
        .expect("failed to create host function: set_input_tensor")
        .with_func::<(i32, i32, i32, i32, i32, i32, i32, i32, i64, i32), ()>("train", train)
        .expect("failed to create set_dataset host function")
        .build(module_name)
        .expect("failed to create import object");

    let boxed_import = Box::new(import);
    let import = Box::leak(boxed_import);

    import.as_raw_ptr() as *mut _
}

/// Defines PluginDescriptor
#[export_name = "WasmEdge_Plugin_GetDescriptor"]
pub extern "C" fn plugin_hook() -> *const ffi::WasmEdge_PluginDescriptor {
    let name = "naive_math_plugin";
    let desc = "this is naive math plugin";
    let version = PluginVersion::new(0, 0, 0, 0);
    let plugin_descriptor = PluginDescriptor::new(name, desc, version)
        .expect("Failed to create plugin descriptor")
        .add_module_descriptor(
            "naive_math_module",
            "this is naive math module",
            Some(create_test_module),
        )
        .expect("Failed to add module descriptor");

    let boxed_plugin = Box::new(plugin_descriptor);
    let plugin = Box::leak(boxed_plugin);

    plugin.as_raw_ptr()
}

pub fn to_tch_tensor(dtype: u8, dims: &[i64], data: &[u8]) -> tch::Tensor {
    match dtype {
        0 => unimplemented!("F16"),
        1 => {
            let data = protocol::bytes_to_f32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        2 => Tensor::of_slice(data).reshape(dims),
        3 => {
            let data = protocol::bytes_to_i32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        _ => panic!("plugin: train_images: unsupported dtype: {dtype}"),
    }
}

pub fn train_model(dataset: Dataset, device: Device) -> Result<()> {
    println!("start training ...");

    let module_path = "/root/workspace/wasi-nn-training/model.pt";

    let vs = VarStore::new(device);
    let mut trainable = TrainableCModule::load(module_path, vs.root())?;
    trainable.set_train();

    println!("*** accuracy");

    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..20 {
        for (images, labels) in dataset
            .train_iter(128)
            .shuffle()
            .to_device(vs.device())
            .take(50)
        {
            let loss = trainable
                .forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
        }
        let test_accuracy = trainable.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            vs.device(),
            1024,
        );
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    trainable.save("trained_model.pt")?;

    println!("[Done]");
    Ok(())
}

// * interface protocol

pub mod protocol {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    pub fn to_bytes<'a, T>(data: &'a [T]) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const _,
                data.len() * std::mem::size_of::<T>(),
            )
        }
    }

    pub fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
        let chunks: Vec<&[u8]> = data.chunks(4).collect();
        let v: Vec<f32> = chunks
            .into_iter()
            .map(|c| {
                let mut rdr = Cursor::new(c);
                rdr.read_f32::<LittleEndian>().expect("failed to read")
            })
            .collect();

        v.into_iter().collect()
    }

    pub fn bytes_to_i32_vec(data: &[u8]) -> Vec<i32> {
        let chunks: Vec<&[u8]> = data.chunks(4).collect();
        let v: Vec<i32> = chunks
            .into_iter()
            .map(|c| {
                let mut rdr = Cursor::new(c);
                rdr.read_i32::<LittleEndian>().expect("failed to read")
            })
            .collect();

        v.into_iter().collect()
    }

    pub fn bytes_to_i64_vec(data: &[u8]) -> Vec<i64> {
        let chunks: Vec<&[u8]> = data.chunks(4).collect();
        let v: Vec<i64> = chunks
            .into_iter()
            .map(|c| {
                let mut rdr = Cursor::new(c);
                rdr.read_i64::<LittleEndian>().expect("failed to read")
            })
            .collect();

        v.into_iter().collect()
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Array {
        pub data: *const u8,
        pub size: i32,
    }

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct MyTensor {
        pub data: *const u8, // 4 bytes
        pub data_size: u32,  // 4 bytes
        pub dims: *const u8, // 4 bytes
        pub dims_size: u32,  // 4 bytes
        pub ty: u8,          // 1 byte
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Tensor<'a> {
        pub dimensions: TensorDimensions<'a>,
        pub type_: TensorType,
        pub data: TensorData<'a>,
    }

    pub type TensorData<'a> = &'a [u8];
    pub type TensorDimensions<'a> = &'a [u32];

    #[repr(transparent)]
    #[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct TensorType(u8);
    pub const TENSOR_TYPE_F16: TensorType = TensorType(0);
    pub const TENSOR_TYPE_F32: TensorType = TensorType(1);
    pub const TENSOR_TYPE_U8: TensorType = TensorType(2);
    pub const TENSOR_TYPE_I32: TensorType = TensorType(3);
    impl TensorType {
        pub const fn raw(&self) -> u8 {
            self.0
        }

        pub fn name(&self) -> &'static str {
            match self.0 {
                0 => "F16",
                1 => "F32",
                2 => "U8",
                3 => "I32",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }
        pub fn message(&self) -> &'static str {
            match self.0 {
                0 => "",
                1 => "",
                2 => "",
                3 => "",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }
    }
    impl std::fmt::Debug for TensorType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TensorType")
                .field("code", &self.0)
                .field("name", &self.name())
                .field("message", &self.message())
                .finish()
        }
    }
}
