def install_dependencies():
    import os
    import subprocess
    import time
    import urllib.request
    import tarfile
    import sys

    def download_file(url, filename):
        print(f"Downloading {filename}...")
        start_time = time.time()
        response = urllib.request.urlopen(url)
        file_size = int(response.headers["Content-Length"])
        downloaded = 0
        block_size = 8192
        with open(filename, "wb") as file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                file.write(buffer)
                downloaded += len(buffer)
                progress = downloaded / file_size * 100
                print(f"Progress: {progress:.2f}%", end="\r")
        print()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Download complete: {filename} (Time: {elapsed_time:.2f} seconds)")

    def extract_tar_gz(filename):
        print(f"Extracting {filename}...")
        start_time = time.time()
        with tarfile.open(filename, "r:gz") as tar:
            file_count = len(tar.getmembers())
            extracted = 0
            for member in tar:
                tar.extract(member)
                extracted += 1
                progress = extracted / file_count * 100
                print(f"Progress: {progress:.2f}%", end="\r")
        print()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Extraction complete: {filename} (Time: {elapsed_time:.2f} seconds)")

    def install_ta_lib():
        print("Installing TA-Lib...")
        start_time = time.time()
        os.chdir("ta-lib")
        subprocess.run(["./configure", "--prefix=/usr"], shell=True)
        subprocess.run(
            ["make", "-s"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["sudo", "make", "-s", "install"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.chdir("..")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"TA-Lib installation complete (Time: {elapsed_time:.2f} seconds)")

    def install_dependencies():
        print("Installing Python dependencies...")
        start_time = time.time()
        packages = [
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow-cpu",
            "matplotlib",
            "ta-lib",
            "optuna",
        ]
        total_packages = len(packages)
        progress = 0
        for package in packages:
            progress += 1
            print(f"Installing {package}... ({progress}/{total_packages})")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", package], check=True
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Python dependencies installation complete (Time: {elapsed_time:.2f} seconds)"
        )

    # if __name__ == "__main__":
    print("Welcome to the SPV4 installation!")
    print("This script will install all the necessary dependencies.\n")

    time.sleep(2)

    download_file(
        "https://deac-fra.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz",
        "ta-lib-0.4.0-src.tar.gz",
    )

    print("Extraction process will begin shortly...")
    print("Please wait while the files are being extracted.")

    extract_tar_gz("ta-lib-0.4.0-src.tar.gz")

    print("Extraction process completed successfully!\n")
    print("TA-Lib installation will now begin.")

    install_ta_lib()

    print("TA-Lib installation completed successfully!\n")

    print("Python dependencies installation will now begin.")

    install_dependencies()

    print("Python dependencies installation completed successfully!\n")
    print("SPV4 installation completed successfully!")

    print("Creating 'data' directory...")
    os.makedirs("data", exist_ok=True)
    # os.makedirs(os.path.join("data", "unrefined"), exist_ok=True)
    os.makedirs(os.path.join("data", "refined"), exist_ok=True)

    # filename = os.path.join("data", "add csvs in this folder.txt")
    # with open(filename, "w") as file:
    #     file.write("This is the 'add csvs in this folder.txt' file.")

    print("'data' directories created successfully!\n")
    print("SPV4 installation completed successfully!")
