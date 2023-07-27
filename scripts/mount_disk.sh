sudo lsblk # list disks that are attached
sudo mkdir -p /mnt/ground-data # making directory
sudo mount -o discard,defaults /dev/sdc /mnt/ground-data # mount disk to new virtual machine at mnt/data
sudo chmod -R 777 /mnt/ground-data # adding permissions for stereo_iti_converted
