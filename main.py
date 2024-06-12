import argparse
import os
import struct
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_HEADER = b'CSVD'
FORMAT_HEADER_SIZE = 16
CHANNELS_NUMBER = 4


def compress(input_file: str, output_file: str, method: str, compression: float):
    img = Image.open(input_file)
    n, m = img.height, img.width
    k = np.floor(n * m / (CHANNELS_NUMBER * compression * (n + m + 1))).astype(np.int32)

    # calculate data of compressed image for each rgb channel
    img_arrays = np.asarray(img)
    compressed_image_data = bytes()
    for i in range(3):
        channel = img_arrays[..., i]

        if method == 'numpy':
            compressed_channel = compress_numpy(channel, k)
        elif method == 'simple':
            compressed_channel = compress_channel_power_simple(channel, k, 50000, 1e-8)
        elif method == 'advanced':
            compressed_channel = compress_channel_block_power(channel, k, 1000, 1e-8)
        else:
            raise Exception('Other methods are not implemented')

        compressed_image_data += compressed_channel

    # write data to output_file
    with open(output_file, 'wb') as f:
        header_data = FORMAT_HEADER + struct.pack('<i', n) + struct.pack('<i', m) + struct.pack('<i', k)
        f.write(header_data)
        f.write(compressed_image_data)


def compress_numpy(channel: np.ndarray, k: int) -> bytes:
    u, s, vt = np.linalg.svd(channel, full_matrices=False)
    data = np.concatenate((u[:, :k].ravel(), s[:k], vt[:k, :].ravel()))
    return data.astype(np.float32).tobytes()


# simple power method
def compress_channel_power_simple(a: np.ndarray, k: int, duration: int, eps: float) -> bytes:
    np.random.seed(0)
    n, m = a.shape
    v = np.random.rand(m)
    v /= np.linalg.norm(v)
    u = np.zeros((n, k))
    sigma = np.zeros(k)
    vt = np.zeros((k, m))

    time_bound = time.time() * 1000 + duration
    for i in range(k):
        ata = np.dot(a.T, a)
        counter = 0
        while time.time() * 1000 < time_bound:
            v_new = np.dot(ata, v)
            v_new /= np.linalg.norm(v_new)
            counter += 1
            if counter % 10 == 0 and np.allclose(v_new, v, eps):
                break
            v = v_new

        eigenvalue = np.dot(np.dot(ata, v), v.T)
        vt[i, :] = v
        u[:, i] = np.dot(a, v) / eigenvalue
        sigma[i] = eigenvalue

        if counter == 0:
            break
        a = a - eigenvalue * np.outer(u[:, i], v)

    data = np.concatenate((u.ravel(), sigma, vt.ravel()))
    return data.astype(np.float32).tobytes()


# Block power method taken from [https://sciendo.com/article/10.1515/auom-2015-0024?content-tab=abstract]
def compress_channel_block_power(a: np.ndarray, k: int, duration: int, eps: float) -> bytes:
    np.random.seed(0)
    n, m = a.shape
    u = np.zeros((n, k))
    sigma = np.zeros(k)
    v = np.zeros((m, k))

    counter = 0
    time_bound = time.time() * 1000 + duration
    while time.time() * 1000 < time_bound:
        q, _ = np.linalg.qr(np.dot(a, v))
        u = q[:, :k]
        q, r = np.linalg.qr(np.dot(a.T, u))
        v = q[:, :k]
        sigma = np.diag(r[:k, :k])
        counter += 1
        if counter % 10 == 0 and np.allclose(np.dot(a, v), np.dot(u, r[:k, :k]), eps):
            break

    data = np.concatenate((u.ravel(), sigma, v.T.ravel()))
    return data.astype(np.float32).tobytes()


def unpack_channel(byte_data, n, m, k) -> np.ndarray:
    split_data = [byte_data[i:i + 4] for i in range(0, len(byte_data), 4)]
    map_obj = map(lambda x: struct.unpack('<f', x), split_data)
    matrix_data = np.array(list(map_obj))
    u = matrix_data[: n * k].reshape(n, k)
    sigma = matrix_data[n * k: n * k + k].ravel()
    vt = matrix_data[n * k + k:].reshape(k, m)
    return np.dot(np.dot(u, np.diag(sigma)), vt)


def decompress(input_file, output_file) -> None:
    with open(input_file, 'rb') as f:
        header_data = f.read(FORMAT_HEADER_SIZE)
        if header_data[:4] != FORMAT_HEADER:
            raise ValueError(f'Incorrect format of {input_file}')

        n = struct.unpack('<i', header_data[4:8])[0]
        m = struct.unpack('<i', header_data[8:12])[0]
        k = struct.unpack('<i', header_data[12:16])[0]
        arrays = [unpack_channel(f.read(4 * k * (n + m + 1)), n, m, k) for _ in range(3)]
        image_matrix = np.stack(arrays, axis=2).clip(0, 255).astype(np.uint8)
        result_image = Image.fromarray(image_matrix)
        result_image.save(output_file)


def main():
    parser = argparse.ArgumentParser(description="Compress or decompress an image.")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True)
    parser.add_argument("--method", type=str, choices=["numpy", "simple", "advanced"])
    parser.add_argument("--compression", type=int)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "compress":
        compress(args.input_file, args.output_file, args.method, args.compression)
    elif args.mode == "decompress":
        decompress(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
