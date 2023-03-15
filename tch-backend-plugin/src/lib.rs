use std::ops::Add;

use tch::Tensor;
use wasmedge_sdk::{
    error::HostFuncError,
    host_function,
    plugin::{ffi, PluginDescriptor, PluginVersion},
    Caller, ImportObjectBuilder, ValType, WasmValue,
};

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

// * interface protocol

mod protocol {

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Array {
        pub data: *const u8,
        pub size: i32,
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
