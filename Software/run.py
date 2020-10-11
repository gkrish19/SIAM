import os

os.system('python top_file.py --model_type=DenseNet-BC --dataset=C100+ --saves --logs --renew-logs --train --test')
os.system('python top_file.py --model_type=DenseNet-BC --dataset=C100+ --saves --logs --renew-logs --train --test --quant --act_width=8 --wgt_width=8')
# os.system('python top_file.py --model_type=VGG19 --dataset=SVHN --saves --logs --renew-logs --vat --train --test --stddevVar=0.1 --quant --act_width=8 --wgt_width=8')
dev = [0.1, 0.2, 0.3, 0.4, 0.5]
adc = [4, 5, 6, 7, 8]
xbar = [64, 128, 256, 512]
for i in dev:
    os.system('python top_file_small.py --model_type=DenseNet-BC --dataset=C100+ --saves --logs --renew-logs --vat --train --test --stddevVar=%.1f --quant --act_width=8 --wgt_width=8' %(i))
    for j in xbar:
        j = int(j)
        for k in adc:
            k= int(k)
            os.system('python top_file_small.py --model_type=DenseNet-BC --dataset=C100+ --saves --logs --renew-logs --vat --train --test --stddevVar=%.1f --quant --act_width=8 --wgt_width=8 --rram --xbar_size=%d --adc_bits=%d' %(i,j,k))
