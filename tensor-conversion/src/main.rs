use mnist::*;
use ndarray::{array, Array2, Array3, ArrayBase};
use std::result::Result;
use std::{error::Error, ops::Div};
use tch::{kind, no_grad, Kind, Tensor};

use std::convert::{TryFrom, TryInto};

const LABELS: i64 = 10; // number of distinct labels
const HEIGHT: usize = 28;
const WIDTH: usize = 28;

const TRAIN_SIZE: usize = 50_000;
const VAL_SIZE: usize = 10_000;
const TEST_SIZE: usize = 10_000;

const N_EPOCHS: i64 = 200;

const THRES: f64 = 0.001;

pub fn image_to_tensor(data: Vec<u8>, dim1: usize, dim2: usize, dim3: usize) -> Tensor {
    // normalize the image as well
    let inp_data: Array3<f32> = Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32 / 256.0);
    // convert to tensor
    let inp_tensor = Tensor::of_slice(inp_data.as_slice().unwrap());
    // reshape so we'll have dim1, dim2*dim3 shape array
    let ax1 = dim1 as i64;
    let ax2 = (dim2 as i64) * (dim3 as i64);
    let shape: Vec<i64> = vec![ax1, ax2];
    let output_data = inp_tensor.reshape(&shape);
    println!("Output image tensor size {:?}", shape);

    output_data
}

fn to_byte_slice<'a>(floats: &'a [f32]) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(floats.as_ptr() as *const _, floats.len() * 4) }
}

fn to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    let mut floats = vec![];
    let len = bytes.len() / 4;
    for i in (0..len).step_by(4) {
        floats.push(f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()));
    }
    floats
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("===== test: converting ndarray -> tch::Tensor -> ndarray");
    {
        let nd_array = array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
        println!("original nd_array:");
        println!("{:?}\n", &nd_array);

        println!("*** converting image to ndarray to tch::Tensor");
        let tensor = Tensor::try_from(nd_array.clone()).unwrap();
        let tensor = tensor.div(2.);
        println!("after divided by 2:");
        println!("{:?}\n", tensor.print());

        println!("*** converting tch::Tensor back to ndarray");
        let nd_array: ndarray::ArrayD<f32> = (&tensor).try_into().unwrap();
        println!("returned nd_array:");
        println!("{:?}", &nd_array);
    }

    println!("\n===== test: converting ndarray -> bytes -> tch::Tensor -> bytes -> ndarray");
    {
        let nd_array: ndarray::Array<f32, _> = array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
        println!("original nd_array:");
        println!("{:?}\n", &nd_array);

        println!("converting ndarry -> bytes");
        let floats = nd_array
            .as_slice()
            .expect("failed to convert ndarray to slice");
        let bytes = to_byte_slice(floats);
        println!("size of bytes: {}", bytes.len());

        println!("converting bytes -> tch::Tensor");
        let tensor = Tensor::of_data_size(bytes, &[3, 3], tch::Kind::Float);
        println!("{:?}\n", tensor.print());

        let tensor = tensor.div(2.);
        println!("after divided by 2:");
        println!("{:?}\n", tensor.print());

        println!("converting tensor -> bytes");
        let data = tensor.data_ptr() as *const u8;
        let mut len = 1;
        for n in tensor.size() {
            len *= n;
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(data, len as usize * std::mem::size_of::<f32>()) };
        println!("size of bytes: {}", bytes.len());

        println!("converting bytes -> ndarray");
        let floats = to_f32_slice(bytes);
        let nd_array: ndarray::Array<f32, _> = ndarray::Array::from_shape_vec((3, 3), floats)?;
        println!("nd_array returned:");
        println!("{:?}\n", &nd_array);
    }

    println!("\n===== test: converting image -> ndarray -> tch::Tensor");
    {
        // Deconstruct the returned Mnist struct.
        let Mnist { trn_img, .. } = MnistBuilder::new()
            .download_and_extract()
            .label_format_digit()
            .training_set_length(TRAIN_SIZE as u32)
            .validation_set_length(VAL_SIZE as u32)
            .test_set_length(TEST_SIZE as u32)
            .finalize();

        println!("trn_img len: {}", trn_img.len());

        let train_data = image_to_tensor(trn_img, TRAIN_SIZE, HEIGHT, WIDTH);

        println!("*** shape of train_data: {:?}", train_data.size());
    }

    Ok(())
}

static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static ROWS: usize = 28;
static COLS: usize = 28;

fn images(path: &std::path::Path, expected_length: u32) -> Vec<u8> {
    use byteorder::{BigEndian, ReadBytesExt};
    use std::io::prelude::*;

    // Read whole file in memory
    let mut content: Vec<u8> = Vec::new();
    let mut file = {
        let mut fh = std::fs::File::open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", path));
        let _ = fh
            .read_to_end(&mut content)
            .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", path.display()));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(
        IMG_MAGIC_NUMBER == magic_number,
        "Expected magic number {} got {}.",
        IMG_MAGIC_NUMBER,
        magic_number
    );
    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
    assert!(
        expected_length == length,
        "Expected data set length of {} got {}.",
        expected_length,
        length
    );
    let rows = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of rows from {:?}.", path))
        as usize;
    assert!(
        ROWS == rows,
        "Expected rows length of {} got {}.",
        ROWS,
        rows
    );
    let cols = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of columns from {:?}.", path))
        as usize;
    assert!(
        COLS == cols,
        "Expected cols length of {} got {}.",
        COLS,
        cols
    );
    // Convert `file` from a Vec to a slice.
    file.to_vec()
}
