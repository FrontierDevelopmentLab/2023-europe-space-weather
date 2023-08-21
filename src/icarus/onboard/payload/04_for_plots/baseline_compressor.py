import glymur

def baseline_compress(data, c=50, tmp_file_name="tmp_000.j2k"):
    # compression_ratios = np.linspace(0, 100, 11)
    #c = 50 # compression ratio ...

    jp2 = glymur.Jp2k(
        tmp_file_name,
        data=data,
        cratios=[c],
        #cratios=[c, c, c/2, c/4, c/8, 1],
    )  # set grayscale J2K

    output = jp2[:]
    path_to_compressed_file = tmp_file_name
    return output, path_to_compressed_file


"""
EXAMPLE USAGE

# 1 load real data
start_io = time.perf_counter()
x_np = load_fits_as_np(input_filename)
end_io = time.perf_counter()
time_io = end_io - start_io
example_input = np.random.rand(1, 3, resolution, resolution)

# 2 baseline method

x_np = (255 * x_np)
print("x_np", x_np.shape, "~", np.min(x_np), np.mean(x_np), np.max(x_np))

data = x_np.astype(np.uint8)[0][0] # one band
print("data", data.shape, "~", np.min(data), np.mean(data), np.max(data))

start = time.perf_counter()
compress(data, c=compression, tmp_file_name=save_folder+"tmp_000.j2k")
end = time.perf_counter()
time_baseline = end - start

print("Success! took:", time_baseline, "sec")

"""