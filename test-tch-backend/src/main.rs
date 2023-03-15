use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

mod plugin {
    #[link(wasm_import_module = "naive-math")]
    extern "C" {
        pub fn add(x: i32, y: i32) -> i32;
        pub fn set_input_tensor(
            data_offset: i32,
            data_size: i32,
            dims_offset: i32,
            dims_size: i32,
            ty: i32,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let x: i32 = args[1].parse().unwrap();
    let y: i32 = args[2].parse().unwrap();

    let res = unsafe { plugin::add(x, y) };
    println!("{x} + {y} = {res}");

    let nd_array: ndarray::Array<f32, _> =
        ndarray::array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
    println!("original nd_array:");
    println!("{:?}\n", &nd_array);

    println!("converting ndarry -> bytes");
    let floats = nd_array
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let bytes = to_byte_slice(floats);
    println!("size of bytes: {}", bytes.len());

    // println!("size of Array: {}", std::mem::size_of::<protocol::Array>());
    // println!("size of *const u8: {}", std::mem::size_of::<*const u8>());
    // println!("size of u8: {}", std::mem::size_of::<u8>());

    // let nums = [1u8, 2, 3];
    // let ptr = &nums as *const u8;
    // println!("ptr address: {:p}", ptr);
    // let offset = ptr as i32;
    // println!("offset: {}", offset);
    // unsafe { plugin::set_input_tensor(offset, 3) };

    // =======

    println!("size of Array: {}", std::mem::size_of::<protocol::Array>());

    let data1 = [1u8, 2, 3, 2, 3, 1, 3, 1, 2];
    let arr1 = protocol::Array {
        data: &data1 as *const _,
        size: data1.len() as i32,
    };
    // let data2 = [6u8, 7, 8];
    // let arr2 = protocol::Array {
    //     data: &data2 as *const _,
    //     size: data2.len() as i32,
    // };

    let arrs = [arr1];
    let ptr = &arrs as *const _ as usize as i32;

    let dims1 = [3u8, 3];
    let dims1_ptr = &dims1 as *const _ as usize as i32;
    let dims1_size = dims1.len();

    unsafe {
        plugin::set_input_tensor(
            ptr,
            (arrs.len() * std::mem::size_of::<protocol::Array>()) as i32,
            dims1_ptr,
            dims1_size as i32,
            2,
        );
    }
}

fn to_byte_slice<'a>(floats: &'a [f32]) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(floats.as_ptr() as *const _, floats.len() * 4) }
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
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

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let sum: f32 = data.iter().sum();
    // log::info!(
    //     "f32_vec_to_bytes: flatten output tensor contains {} elements with sum {}",
    //     data.len(),
    //     sum
    // );
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let result: Vec<u8> = chunks.iter().flatten().copied().collect();

    // log::info!(
    //     "f32_vec_to_bytes: flatten byte output tensor contains {} elements",
    //     result.len()
    // );
    result
}

// * interface protocol

mod protocol {

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct Array {
        pub data: *const u8, // 4 bytes
        pub size: i32,       // 4 bytes
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Tensor<'a> {
        pub dimensions: TensorDimensions<'a>, // 8 bytes
        pub ty: TensorType,                   // 1 bytes
        pub data: TensorData<'a>,             // 8 bytes
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
