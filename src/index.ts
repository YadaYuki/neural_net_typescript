import { TwoLayerNet } from './networks/twoLayerNeuralnet';
import { loadMnist } from './data/load-mnist';
import { choice, getBatchData, maxIdx } from './utils';
import { SimpleConvNet } from './networks/simpleConvNet';

const trainTwoLayerNeuralNet = async () => {
  console.log('Learn TwoLayer Neural Network...');

  const { xTrain, yTrain, yTest, xTest } = await loadMnist('array');
  const network = new TwoLayerNet(784, 50, 10);
  const trainNum = xTrain.shape[0];
  const batchSize = 100;
  const iterNums = 1000;
  const learningRate = 0.1;
  for (let i = 0; i < iterNums; i++) {
    const batchIdxList = choice(trainNum, batchSize);
    const xBatch = getBatchData(batchIdxList, xTrain);
    const yBatch = getBatchData(batchIdxList, yTrain);
    const loss = network.forward(xBatch, yBatch);
    network.backward(); // 逆伝搬
    network.update(learningRate); // パラメータの更新
    if (i % 100 === 0) {
      console.log(`iteration: ${i + 1}`);
      console.log(loss); // 誤差関数の出力結果を表示.
      console.log();
    }
  }
  // テストデータを対象に精度を計算する。
  const testNum = yTest.shape[0];
  const testBatchIdxList = choice(testNum, 1000); // 1000個のデータで精度を検証する。
  const xTestBatch = getBatchData(testBatchIdxList, xTest);
  const yTestBatch = getBatchData(testBatchIdxList, yTest);
  const yPredict = network.predictBatch(xTestBatch);
  const yTestBatchList = yTestBatch.tolist();
  const yPredictList = yPredict.tolist();
  let accurateNum = 0;
  yTestBatchList.map((_, idx) => {
    accurateNum += Number(
      maxIdx(yTestBatchList[idx]) === maxIdx(yPredictList[idx])
    );
  });
  console.log(
    `Accuracy for 1000 of  test data : ${(accurateNum / 1000) * 100} % `
  );
};

const trainCnn = async () => {
  console.log('Learn Convolutional Neural Network...');
  const { xTrain, yTrain } = await loadMnist('image');
  const network = new SimpleConvNet();
  const trainNum = xTrain.shape[0];
  const batchSize = 100;
  const iterNums = 1000;
  const learningRate = 0.1;
  let from = Date.now();
  for (let i = 0; i < iterNums; i++) {
    const batchIdxList = choice(trainNum, batchSize);
    const xBatch = getBatchData(batchIdxList, xTrain);
    const yBatch = getBatchData(batchIdxList, yTrain);
    const loss = network.forward(xBatch, yBatch);
    network.backward(); // 逆伝搬
    network.update(learningRate); // パラメータの更新
    if (i % 100 === 0) {
      const to = Date.now();
      console.log(`iteration: ${i + 1}`);
      console.log(loss); // 誤差関数の出力結果を表示.
      console.log(to - from);
      from = to;
    }
  }
};

type NetworkType = 'cnn' | 'tnn';

const main = () => {
  const arg = process.argv;
  const networkType: NetworkType | string | undefined = arg.pop();
  switch (networkType) {
    case 'cnn':
      trainCnn();
      break;
    case 'tnn':
      trainTwoLayerNeuralNet();
      break;
    default:
      trainTwoLayerNeuralNet();
      break;
  }
};

main();
