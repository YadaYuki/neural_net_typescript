# Neural Network by Typescript :stuck_out_tongue_winking_eye:

## Overview

- Very Simple Neural Network Implementation By Typescript.
-  The architecture and implementation are strongly inspired by **[「Deep learning from scratch」]((https://www.amazon.co.jp/%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8BDeep-Learning-%E2%80%95Python%E3%81%A7%E5%AD%A6%E3%81%B6%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E5%AE%9F%E8%A3%85-%E6%96%8E%E8%97%A4-%E5%BA%B7%E6%AF%85/dp/4873117585))**
- Dataset: MNIST ( http://yann.lecun.com/exdb/mnist/ ) 
- The accuracy is about 95% (batch_size = 100,iteration=10000)

## Architecture of Neuralnet

![image](https://user-images.githubusercontent.com/57289763/132112979-2100d169-4fea-4d43-8d62-cac20570ac8f.png)


<!--
 Figure:
 https://app.diagrams.net/#G1JscsI7Qq8UFcY336XNlRxV86kje2HsxR
-->

## How to Work

### Environment
- yarn=1.22.11
- node=12.15.0
### Setup
```

# Install dependencies
$ yarn install

# Download MNIST
$ yarn download:mnist
```

### Train and Evaluate Model

You can train model by runnning:

```
$ yarn dev
```

Output

```
$ yarn dev

iteration:1
2.301157570863233 # loss

iteration:101
1.6163854318773305

iteration:201
0.6387995662968232

iteration:301
0.4657332402247479

iteration:401
0.470693806907251

iteration:501
0.3244541415721461

iteration:601
0.27824362472848635

...
```

※ It takes about **2 minutes per 100 iteration**, so if iteration num is 10000 , it takes about 200 minutes (= 3h20m). You can change the iteration num by changing iterNums(default: 1000).


### Reference
- [「ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装」](https://www.amazon.co.jp/%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8BDeep-Learning-%E2%80%95Python%E3%81%A7%E5%AD%A6%E3%81%B6%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E5%AE%9F%E8%A3%85-%E6%96%8E%E8%97%A4-%E5%BA%B7%E6%AF%85/dp/4873117585)


