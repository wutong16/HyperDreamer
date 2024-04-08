echo "----------------------- Downloading processed CelebA-HQ -----------------------"

scriptdir=$(dirname $0)
basedir=$(dirname $scriptdir)

dataset_dir=datasets
celebahq_dir=$dataset_dir/celebahq

images_link=""
unsup3d_link=""
extracted_link=""

images_file="celebahq_imgs_crpped.tar"
unsup3d_file="celebahq_unsup3d.tar"
extracted_file="celebahq_extracted.tar"

cd $basedir || exit
echo Operating in \"$(pwd)\".

echo Creating directories.
mkdir -p $celebahq_dir

echo Downloading $images_file from "$images_link".
wget -O $celebahq_dir/$images_file "$images_link"

echo Downloading $unsup3d_file from "$unsup3d_link".
wget -O $celebahq_dir/$unsup3d_file "$unsup3d_link"

echo Downloading $extracted_file from "$extracted_link".
wget -O $celebahq_dir/$extracted_file "$extracted_link"

echo Extracting $images_file.
tar -C $celebahq_dir -xf $celebahq_dir/$images_file

echo Extracting $unsup3d_file.
tar -C $celebahq_dir -xf $celebahq_dir/$unsup3d_file

echo Extracting $extracted_file.
tar -C $celebahq_dir -xf $celebahq_dir/$extracted_file

echo "Extraction complete."
echo "Feel free to remove the archive files in $celebahq_dir: $images_file, $unsup3d_file, $extracted_file"
