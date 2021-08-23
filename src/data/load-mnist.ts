import nj from 'numjs';
import fetch from 'node-fetch';
import fs from 'fs';
import zlib from 'zlib';
import path from 'path';

const keyFiles = {
  trainImg: 'train-images-idx3-ubyte',
  trainLabel: 'train-labels-idx1-ubyte',
  testImg: 't10k-images-idx3-ubyte',
  testLabel: 't10k-labels-idx1-ubyte',
} as const;
type FileKey = keyof typeof keyFiles;

const baseUrl = 'http://yann.lecun.com/exdb/mnist';

const donwloadMnist = async (): Promise<void> => {
  console.log(`Download MNIST`);
  for (const key in keyFiles) {
    console.log(`Downloading ...${key}`);
    const data = await fetch(`${baseUrl}/${keyFiles[key as FileKey]}.gz`, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
      },
    });
    const gzipedData = await data.buffer();
    zlib.gunzip(gzipedData, (err, gunzipedData) => {
      if (err) throw err;
      fs.writeFileSync(
        path.join(__dirname, 'mnist', keyFiles[key as FileKey]),
        gunzipedData
      );
    });
  }
};

const _loadLabelData = async (labelFilename: string): Promise<nj.NdArray> => {
  const offset = 8;
  let labelData = fs.readFileSync(path.join(__dirname, 'mnist', labelFilename));
  labelData = labelData.slice(offset);
  const label = nj.array(Array.from(labelData));
  return label;
};

const loadLabelData = async (): Promise<{
  trainLabel: nj.NdArray;
  testLabel: nj.NdArray;
}> => {
  const filenameDict = {
    trainLabelFilename: 'train-labels-idx1-ubyte',
    testLabelFilename: 't10k-labels-idx1-ubyte',
  };
  const trainLabel = await _loadLabelData(filenameDict.trainLabelFilename);
  const testLabel = await _loadLabelData(filenameDict.testLabelFilename);
  return {
    trainLabel,
    testLabel,
  };
};

const _loadImageData = async (imgFilename: string): Promise<nj.NdArray> => {
  const offset = 16;
  const mnistDataSize = 784; // 28 * 28
  let imgData = fs.readFileSync(path.join(__dirname, 'mnist', imgFilename));
  const dataNum = parseInt(imgData.slice(4, 8).toString('hex'), 16);
  imgData = imgData.slice(offset);
  const img = nj.array(Array.from(imgData)).reshape(dataNum, mnistDataSize);
  return img;
};

/*
loadImageDataはmnistの画像データを読み込み、784(=28*28)個の要素を持つ一次元のnj.Ndarrayに変換し、それらを値として返す。
*/
const loadImageData = async (): Promise<{
  trainImg: nj.NdArray;
  testImg: nj.NdArray;
}> => {
  const filenameDict = {
    trainImgFilename: 'train-images-idx3-ubyte',
    testImgFilename: 't10k-images-idx3-ubyte',
  };
  const trainImg = await _loadImageData(filenameDict.trainImgFilename);
  const testImg = await _loadImageData(filenameDict.testImgFilename);
  return {
    trainImg,
    testImg,
  };
};

export const loadMnist = async (): Promise<{
  xTrain: nj.NdArray;
  yTrain: nj.NdArray;
  xTest: nj.NdArray;
  yTest: nj.NdArray;
}> => {
  const filenameArr = Object.values(keyFiles);
  try {
    for (const filename of filenameArr) {
      fs.accessSync(path.join(__dirname, 'mnist', filename));
    }
  } catch (e) {
    await donwloadMnist();
  }
  // load mnist from dir

  const { trainLabel, testLabel } = await loadLabelData();
  const { trainImg, testImg } = await loadImageData();
  return {
    xTrain: trainImg,
    yTrain: trainLabel,
    xTest: testImg,
    yTest: testLabel,
  };
};
