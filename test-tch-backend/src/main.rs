// use mnist::*;
use ndarray::{Array2, Array3};
use std::fs::File;
use std::io::{self, BufReader, Read, Result, Write};
use std::path::Path;

mod plugin {
    #[link(wasm_import_module = "naive-math")]
    extern "C" {
        pub fn train(
            train_images_offset: i32,
            train_images_size: i32,
            train_labels_offset: i32,
            train_labels_size: i32,
            test_images_offset: i32,
            test_images_size: i32,
            test_labels_offset: i32,
            test_labels_size: i32,
            labels: i64,
            device: i32,
        );
    }
}

fn main() {
    // training images
    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();
    let trn_img_filename = Path::new("data/train-images-idx3-ubyte");
    let train_images = read_images(trn_img_filename).expect("failed to load training images");
    let train_images = train_images
        .into_shape((60_000, 1, 28, 28))
        .expect("failed to reshape train_images");
    let train_images_dims: Vec<i32> = train_images.shape().iter().map(|&c| c as i32).collect();
    let train_images_dims_bytes = protocol::to_bytes(train_images_dims.as_slice());
    let train_images_slice = train_images
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let train_images_bytes = protocol::to_bytes(train_images_slice);
    // println!("size of train_images_bytes: {}", train_images_bytes.len());
    println!("[Done] (shape: {:?}, dtype: f32)", train_images.shape());

    // training labels
    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();
    let trn_lbl_filename = Path::new("data/train-labels-idx1-ubyte");
    let train_labels = read_labels(trn_lbl_filename).expect("failed to load training lables");
    let train_labels_dims: Vec<i32> = train_labels.shape().iter().map(|&c| c as i32).collect();
    let train_labels_dims_bytes = protocol::to_bytes(train_labels_dims.as_slice());
    let train_labels_slice = train_labels
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let train_labels_bytes = protocol::to_bytes(train_labels_slice);
    // println!("size of train_labels_bytes: {}", train_labels_bytes.len());
    println!("[Done] (shape: {:?}, dtype: i64)", train_labels.shape());

    // test images
    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();
    let tst_img_filename = Path::new("data/t10k-images-idx3-ubyte");
    let test_images = read_images(tst_img_filename).expect("failed to load test images");
    let test_images = test_images
        .into_shape((10_000, 1, 28, 28))
        .expect("failed to reshape test_images");
    let test_images_dims: Vec<i32> = test_images.shape().iter().map(|&c| c as i32).collect();
    let test_images_dims_bytes = protocol::to_bytes(test_images_dims.as_slice());
    let test_images_slice = test_images
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let test_images_bytes = protocol::to_bytes(test_images_slice);
    // println!("size of test_images_bytes: {}", test_images_bytes.len());
    println!("[Done] (shape: {:?}, dtype: f32)", test_images.shape());

    // test labels
    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();
    let tst_lbl_filename = Path::new("data/t10k-labels-idx1-ubyte");
    let test_labels = read_labels(tst_lbl_filename).expect("failed to load test labels");
    let test_labels_dims: Vec<i32> = test_labels.shape().iter().map(|&c| c as i32).collect();
    let test_labels_dims_bytes = protocol::to_bytes(test_labels_dims.as_slice());
    let test_labels_slice = test_labels
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let test_labels_bytes = protocol::to_bytes(test_labels_slice);
    // println!("size of test_labels_bytes: {}", test_labels_bytes.len());
    println!("[Done] (shape: {:?}, dtype: i64) ", test_labels.shape());

    let train_images = protocol::MyTensor {
        data: train_images_bytes.as_ptr() as *const _,
        data_size: train_images_bytes.len() as u32,
        dims: train_images_dims_bytes.as_ptr() as *const _,
        dims_size: train_images_dims_bytes.len() as u32,
        ty: 1, // f32
    };
    let train_images_offset = &train_images as *const _ as usize as i32;
    let train_images_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let train_labels = protocol::MyTensor {
        data: train_labels_bytes.as_ptr() as *const _,
        data_size: train_labels_bytes.len() as u32,
        dims: train_labels_dims_bytes.as_ptr() as *const _,
        dims_size: train_labels_dims_bytes.len() as u32,
        ty: 4, // i64
    };
    let train_labels_offset = &train_labels as *const _ as usize as i32;
    let train_labels_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let test_images = protocol::MyTensor {
        data: test_images_bytes.as_ptr() as *const _,
        data_size: test_images_bytes.len() as u32,
        dims: test_images_dims_bytes.as_ptr() as *const _,
        dims_size: test_images_dims_bytes.len() as u32,
        ty: 1, // f32
    };
    let test_images_offset = &test_images as *const _ as usize as i32;
    let test_images_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let test_labels = protocol::MyTensor {
        data: test_labels_bytes.as_ptr() as *const _,
        data_size: test_labels_bytes.len() as u32,
        dims: test_labels_dims_bytes.as_ptr() as *const _,
        dims_size: test_labels_dims_bytes.len() as u32,
        ty: 4, // i64
    };
    let test_labels_offset = &test_labels as *const _ as usize as i32;
    let test_labels_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    unsafe {
        plugin::train(
            train_images_offset,
            train_images_size,
            train_labels_offset,
            train_labels_size,
            test_images_offset,
            test_images_size,
            test_labels_offset,
            test_labels_size,
            10,
            0, // device: CPU
        )
    }
}

pub fn image_to_ndarray(data: Vec<u8>, dim1: usize, dim2: usize, dim3: usize) -> Array3<f32> {
    // normalize the image as well
    let inp_data: Array3<f32> = Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32 / 256.0);

    inp_data
}

pub fn labels_to_ndarray(data: Vec<u8>, dim1: usize, dim2: usize) -> Array2<i64> {
    let inp_data: Array2<i64> = Array2::from_shape_vec((dim1, dim2), data)
        .expect("Error converting data to 2D array")
        .map(|x| *x as i64);

    inp_data
}

fn read_u32<T: Read>(reader: &mut T) -> Result<u32> {
    let mut b = vec![0u8; 4];
    reader.read_exact(&mut b)?;
    let (result, _) = b.iter().rev().fold((0u64, 1u64), |(s, basis), &x| {
        (s + basis * u64::from(x), basis * 256)
    });
    Ok(result as u32)
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("incorrect magic number {magic_number} != {expected}"),
        ));
    }
    Ok(())
}

use std::convert::From;

fn read_labels(filename: &std::path::Path) -> Result<ndarray::Array<i64, ndarray::Ix1>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;

    Ok(ndarray::Array::from_vec(data).mapv(i64::from))

    // Ok(Tensor::of_slice(&data).to_kind(Kind::Int64))
}

fn read_images(filename: &std::path::Path) -> Result<ndarray::Array<f32, ndarray::Ix2>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)?;
    let rows = read_u32(&mut buf_reader)?;
    let cols = read_u32(&mut buf_reader)?;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len as usize];
    buf_reader.read_exact(&mut data)?;

    let shape = (samples as usize, (rows * cols) as usize);
    let arr = ndarray::Array::from_shape_vec(shape, data)
        .expect("failed to create ndarray for images")
        .mapv(f32::from)
        / 255.;

    Ok(arr)

    // let tensor = Tensor::of_slice(&data)
    //     .view((i64::from(samples), i64::from(rows * cols)))
    //     .to_kind(Kind::Float);
    // Ok(tensor / 255.)
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
        let chunks: Vec<&[u8]> = data.chunks(8).collect();
        let v: Vec<i64> = chunks
            .into_iter()
            .map(|c| {
                let mut rdr = Cursor::new(c);
                rdr.read_i64::<LittleEndian>().expect(
                    format!(
                        "plugin: protocol: failed to read. input data size: {}",
                        data.len()
                    )
                    .as_str(),
                )
            })
            .collect();

        v.into_iter().collect()
    }
    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct Array {
        pub data: *const u8, // 4 bytes
        pub size: i32,       // 4 bytes
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

    pub enum Device {
        Cpu,
        Cuda(usize),
        Mps,
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
}
