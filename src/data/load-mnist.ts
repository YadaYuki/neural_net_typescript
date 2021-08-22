import nj from 'numjs';
import fetch from 'node-fetch';
import fs from 'fs';

const keyFile = {
  trainImg: 'train-images-idx3-ubyte.gz',
  trainLabel: 'train-labels-idx1-ubyte.gz',
  testImg: 't10k-images-idx3-ubyte.gz',
  testLabel: 't10k-labels-idx1-ubyte.gz',
};

const baseUrl = 'http://yann.lecun.com/exdb/mnist';

type ArrayLengthMutationKeys = 'splice' | 'push' | 'pop' | 'shift' | 'unshift';

type FixedLengthArray<T, L extends number, TObj = [T, ...Array<T>]> = Pick<
  TObj,
  Exclude<keyof TObj, ArrayLengthMutationKeys>
> & {
  readonly length: L;
  [I: number]: T;
  [Symbol.iterator]: () => IterableIterator<T>;
};

export const loadMnist = async (): Promise<FixedLengthArray<nj.NdArray, 4>> => {
  // download if file not exist
  const data = await fetch(`${baseUrl}/${keyFile.trainImg}`, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    },
  });
  const buf = await data.buffer();
  fs.writeFile(keyFile.trainImg, buf, (err) => {
    if (err) throw err;
    console.log('正常に書き込みが完了しました');
  });
  // image file to ndarray
  return [
    nj.zeros([3, 3, 3], 'uint8'),
    nj.zeros([3, 3, 3], 'uint8'),
    nj.zeros([3, 3, 3], 'uint8'),
    nj.zeros([3, 3, 3], 'uint8'),
  ];
};
