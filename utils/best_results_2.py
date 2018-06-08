import glob

if __name__ == '__main__':
    names = glob.glob("../results/**/**/*.log")
    for name in names:
        print(name)
        splitted = name.split('/')
        network = splitted[-2]
        dataset = splitted[-3]
        loss = splitted[-1].replace('.log', '')
        with open(name, mode='r') as f:
            train, val, test = None, None, None
            for line in f.readlines():
                try:
                    if not 'Best' in line:
                        ind = line.find('{')
                        j = line[ind:]
                        print(eval(j)['val_acc_metric'])
                        if 'Train' in line:
                            train = eval(j)['val_acc_metric']
                        if 'Val' in line:
                            val = eval(j)['val_acc_metric']
                        if 'Test' in line:
                            test = eval(j)['val_acc_metric']
                except Exception as e:
                    print(e)
            with open('res.txt', mode='a') as resf:
                resf.write('%s %s %s %s %s %s\n' % (dataset, loss, network, train, val, test))
