sudo lsblk #list disks that are attached
sudo mkdir -p /mnt/onboard_data #making directory
sudo mount -o discard,defaults /dev/sdb /mnt/onboard_data #mount disk to new virtual machine at mnt/data
sudo chmod -R 777 /mnt/onboard_data #adding permissions
