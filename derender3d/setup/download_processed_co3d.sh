echo "----------------------- Downloading processed Co3D -----------------------"


read -p "Please make yourself familiar with the Co3D license at https://github.com/facebookresearch/co3d.
I agree to this license [yY|nN]. " -n 1 -r
echo #
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


scriptdir=$(dirname $0)
basedir=$(dirname $scriptdir)

dataset_dir=datasets
co3d_dir=$dataset_dir/co3d

template_link="https://www.robots.ox.ac.uk/~vgg/research/derender3d/data/extracted_%s.tar"
template_file="processed_%s.tar"
categories=("hydrant" "toybus" "toytruck" "toyplane" "bench" "chair" "sandwich" "toilet" "umbrella")

cd $basedir || exit
echo Operating in \"$(pwd)\".

echo Creating directories.
mkdir -p $co3d_dir

for category in "${categories[@]}"
do
  category_link=$(printf $template_link $category)
  category_file=$(printf $template_file $category)
  echo Downloading archive from $category_link to $category_file.
  wget -O $co3d_dir/$category_file $category_link
done

for category in "${categories[@]}"
do
  category_file=$(printf $template_file $category)
  echo Extracting $category_file.
  tar -C $co3d_dir/$co3d_dir $category_file
done

echo "Extraction complete."
echo "Feel free to remove the archive files in $co3d_dir: ${categories[*]// /|}"
