import conifer
import plotting


#Covert model to FPGA firmware with `conifer`
cfg = conifer.backends.xilinxhls.auto_config()

# print the config
print('Default Configuration\n' + '-' * 50)
plotting.print_dict(cfg)
print('-' * 50)

# modify the config
cfg['OutputDir'] = '/eos/user/s/swaldych/smart_pix/labels/generated_firmware_files' #where to put all generated firmware files
cfg['XilinxPart'] = 'xcu250-figd2104-2L-e' #the part number for an FPGA. Taken from example (Alveo U50)

# print the config again
print('Modified Configuration\n' + '-' * 50)
plotting.print_dict(cfg)
print('-' * 50)
