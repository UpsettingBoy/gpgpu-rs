# -*- mode: ruby -*-
# vi: set ft=ruby :
Vagrant.configure("2") do |config|
  config.vm.box = "generic/ubuntu2104"
  
  config.vm.synced_folder ".", "/gpgpu", type: "smb"
  config.vm.network :private_network, ip: "192.168.33.10"

  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get upgrade -y
    apt-get install -y build-essential libgl1 freeglut3-dev xvfb
  SHELL

  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    curl https://sh.rustup.rs -sSf | sh -s -- -y
  SHELL

  config.vm.provider "hyperv" do |vb|
    vb.memory = "1536"
    vb.cpus = 2
  end
end