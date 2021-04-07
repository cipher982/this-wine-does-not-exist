$py_file = "C:\Users\david\Documents\github\stylegan2-ada-pytorch\generate.py"
$model_pkl = "D:\data\stylegan2_ada_outputs\training-runs_512\00011-labels_clean-cond-stylegan2-batch8-noaug-resumecustom\network-snapshot-009480.pkl"
$samples = 1000


for ($class = 0 ; $class -le 15 ; $class++)
{
  python $py_file --network $model_pkl --outdir "H:\data\stylegan2_ada_outputs\outputs_00011-9480_$class" --seeds 0-$samples --class $class
}