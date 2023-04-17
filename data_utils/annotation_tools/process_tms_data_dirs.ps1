# this scripts batch process the json annotation files for tms keypoint annotations
$dirs = "C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\304 - 1802 3A",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\CJ-335\335 - 1706 Niv3",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\Test bleu poutrelle 1 - 14 mars 2023\Sans soudure",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\Test bleu poutrelle 1 - 14 mars 2023\Sans soudure",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\Test bleu poutrelle 2 - 14 mars 2023\Avec crayon bleu",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\Test bleu poutrelle 2 - 14 mars 2023\Sans crayon bleu",
"C:\Users\LTI\OneDrive - LTI\Documents\LTI\Clients\TMS systems\data\Test Crayon"

$destination_dir = "C:\repos\Pointnet_Pointnet2_pytorch\data\poutrelle\p1"
for($i=0; $i -lt $dirs.Length; $i++){
    $src = $dirs[$i]
    Write-Output "Processing file in $src`n"
    #C:\Users\LTI\anaconda3\envs\labelCloud_env\python.exe C:\repos\tms-systems-poc\annotations_tools\create_3D_keypoints_annotations.py --source_dir $src_dir
    $src = $dirs[$i] + "\keypoints_ply_files\*"
    Write-Output "Copying files from $src to $destination_dir`n"
    Copy-Item -Force -Path $src -Destination $destination_dir
}
