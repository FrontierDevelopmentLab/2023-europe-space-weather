sudo lsblk #list disks that are attached
sudo mkdir -p /mnt/nerf-data #making directory
sudo mount -o discard,defaults /dev/sdb /mnt/nerf-data #mount disk to new virtual machine at mnt/data
sudo chmod -R 777 /mnt/nerf-data #adding permissions for stereo_iti_converted
