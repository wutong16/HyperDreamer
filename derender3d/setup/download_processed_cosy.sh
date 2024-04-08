echo "----------------------- Downloading processed COSy -----------------------"

read -p "COSy is provided under the CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/3.0/).
I agree to this license [yY|nN]. " -n 1 -r
echo #
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

scriptdir=$(dirname $0)
basedir=$(dirname $scriptdir)

dataset_dir=datasets
cosy_dir=$dataset_dir/cosy

cosy_link="https://www.robots.ox.ac.uk/~vgg/research/derender3d/data/COSy.tar"

cosy_file="cosy.tar"

cd $basedir || exit
echo Operating in \"$(pwd)\".

echo Creating directories.
mkdir -p $cosy_dir

echo Downloading $cosy_file from "$cosy_link".
wget -O $cosy_dir/$cosy_file "$cosy_link"

echo Extracting $cosy_file.
tar -C $cosy_dir -xf $cosy_dir/$cosy_file

echo "Extraction complete."
echo "Feel free to remove the archive files in $cosy_dir: $cosy_file"
