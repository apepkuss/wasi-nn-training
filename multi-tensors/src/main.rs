mod plugin {
    #[link(wasm_import_module = "naive-math")]
    extern "C" {
        pub fn set_input_tensor(offset: i32, size: i32, x: i32, y: i32, z: i32);
    }
}

use std::mem;

fn main() {
    let data1: GraphBuilder<'_> = &[1_u8, 2, 3, 4, 5];
    println!("data1 size: {}", mem::size_of::<GraphBuilder<'_>>());
    // let data1_offset = &data1 as *const _ as i32;
    // let data1_size = data1.len() as i32;

    println!("size of *const u8: {}", mem::size_of::<*const u8>());
    println!("size of GraphBuilder: {}", mem::size_of::<GraphBuilder>());
    println!(
        "size of GraphBuilderArray: {}",
        mem::size_of::<GraphBuilderArray>()
    );

    let data2: GraphBuilder<'_> = &[5_u8, 4, 3, 2, 1];

    let data: GraphBuilderArray<'_> = &[data1, data2];
    println!("data size: {}", mem::size_of::<GraphBuilderArray>());
    println!("data ptr: {:p}", data.as_ptr());
    let data_offset = data.as_ptr() as i32;
    let data_size = data.len() as i32;
    println!("data_offset: {data_offset}, data_size: {data_size}");

    // =======

    let slice1 = &[1_u8, 2, 3, 2, 3, 1];
    let dim1 = &[1024_i32, 1024];
    let dim1_bytes = to_bytes(dim1);
    let tensor1 = Tensor {
        dimensions: dim1_bytes,
        ty: TENSOR_TYPE_I64,
        data: slice1,
    };
    println!("dims: {:?}", tensor1.dimensions);
    println!("size of Tensor: {}", mem::size_of::<Tensor>());
    println!(
        "size of TensorDimensions: {}",
        mem::size_of::<TensorDimensions>()
    );
    println!("size of TensorType: {}", mem::size_of::<TensorType>());
    println!("size of TensorData: {}", mem::size_of::<TensorData>());
    let ptr_tensor1 = &tensor1 as *const _;
    println!("tensor1 ptr: {:p}", ptr_tensor1);
    let offset_tensor1 = ptr_tensor1 as usize as i32;
    println!("tensor1 offset: {}", offset_tensor1);

    let slice2 = &[1_u8, 2, 3, 4];
    let dim2 = &[2_i32, 2];
    let dim2_bytes = to_bytes(dim2);
    let tensor2 = Tensor {
        dimensions: dim2_bytes,
        ty: TENSOR_TYPE_F32,
        data: slice2,
    };

    println!("size of TensorElement: {}", mem::size_of::<TensorElement>());
    println!("size of TensorArray: {}", mem::size_of::<TensorArray>());

    println!("addr: {:p}", &tensor1);

    let tensors: TensorArray<'_> = &[&tensor1, &tensor2];
    let offset_tensors = tensors.as_ptr() as i32;
    let len_tensors = tensors.len() as i32;

    unsafe {
        plugin::set_input_tensor(data_offset, data_size, offset_tensors, len_tensors, 0);
    }
}

pub fn to_bytes<'a, T>(data: &'a [T]) -> &'a [u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const _,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}

pub type GraphBuilder<'a> = &'a [u8];
pub type GraphBuilderArray<'a> = &'a [GraphBuilder<'a>];

pub type TensorElement<'a> = &'a Tensor<'a>;
pub type TensorArray<'a> = &'a [TensorElement<'a>];

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Tensor<'a> {
    pub dimensions: TensorDimensions<'a>, // 8 bytes
    pub data: TensorData<'a>,             // 8 bytes
    pub ty: TensorType,                   // 1 bytes
}

pub type TensorData<'a> = &'a [u8];
pub type TensorDimensions<'a> = &'a [u8];

#[repr(transparent)]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TensorType(u8);
pub const TENSOR_TYPE_F16: TensorType = TensorType(0);
pub const TENSOR_TYPE_F32: TensorType = TensorType(1);
pub const TENSOR_TYPE_U8: TensorType = TensorType(2);
pub const TENSOR_TYPE_I32: TensorType = TensorType(3);
pub const TENSOR_TYPE_I64: TensorType = TensorType(4);
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
