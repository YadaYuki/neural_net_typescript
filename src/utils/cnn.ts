import nj from 'numjs';

/**
 * 画像のバッチデータや複数のcnnの重み行列といった4次元データを2次元の行列に変換する。
 */
export const im2col = (
  input: nj.NdArray<number[][][]>,
  filterH: number,
  filterW: number,
  stride = 1,
  padding = 0
): nj.NdArray<number[]> => {
  const inputShape = input.shape;
  const d = inputShape[1];
  const h = inputShape[2];
  const w = inputShape[3];
  const inputList = input.tolist();
  const colItem: number[][] = [];
  inputList.forEach((inputItem) => {
    for (let i = 0; i < h - filterH + 1; i++) {
      for (let j = 0; j < w - filterW + 1; j++) {
        const data: number[] = [];
        for (let k = 0; k < d; k++) {
          const inputDataCol = inputItem[k];
          const windowArr = inputDataCol.slice(i, i + filterH);
          windowArr.forEach((item) => {
            for (let l = j; l < j + filterW; l++) {
              data.push(item[l]);
            }
          });
        }
        colItem.push(data);
      }
    }
  });
  return nj.array(colItem);
};

/**
 * 2次元の行列を画像のバッチデータや複数のcnnの重み行列といった4次元データに変換する。im2colと逆の効果を持つ。
 */
export const col2im = (
  input: nj.NdArray<number[]>,
  shape: { n: number; d: number; h: number; w: number }, // TODO:migrate to fixed length array
  filterH: number,
  filterW: number,
  stride = 1,
  padding = 0
): nj.NdArray<number[][][]> => {
  const { n: N, d: D, h: H, w: W } = shape;
  const OH = (H + 2 * padding - filterH) / stride + 1;
  const OW = (W + 2 * padding - filterW) / stride + 1;
  const filterSize = filterH * filterW;
  const filterOutputSize = OH * OW;
  const imgArr: number[][][][][] = []; // (n,d,h,w)
  for (let i = 0; i < N * filterOutputSize; i += filterOutputSize) {
    const imgData: number[][][][] = []; // (d,h,w) single image
    for (let j = 0; j < D * filterSize; j = j + filterSize) {
      imgData.push(
        (
          input
            .slice([i, i + filterOutputSize], [j, j + filterSize])
            .reshape(OH * OW, filterH, filterW) as nj.NdArray<number[][]>
        ).tolist()
      );
    }
    imgArr.push(imgData);
  }
  const col = (
    nj.zeros([N, D, H, W]).reshape(N, D, H, W) as nj.NdArray<number[][][]>
  ).tolist();
  for (let g = 0; g < N; g++) {
    for (let h = 0; h < D; h++) {
      let imgArrIdx = 0;
      for (let i = 0; i < H - filterH + 1; i = i + stride) {
        for (let j = 0; j < W - filterW + 1; j = j + stride) {
          for (let k = 0; k < filterH; k++) {
            for (let l = 0; l < filterW; l++) {
              col[g][h][i + k][j + l] += imgArr[g][h][imgArrIdx][k][l];
            }
          }
          imgArrIdx++;
        }
      }
    }
  }

  return nj.array(col);
};
