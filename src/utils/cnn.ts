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
  const n = inputShape[0];
  const d = inputShape[1];
  const h = inputShape[2];
  const w = inputShape[3];
  // const outputW = filterH * filterW * d;
  // const filterOutH = (h + 2 * padding - filterH) / stride + 1;
  // const filterOutW = (w + 2 * padding - filterW) / stride + 1;
  // // const outputH = n * filterOutW * filterOutH;
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
  const { n, d, h, w } = shape;
  const OH = h + 2 * padding - filterH / stride + 1;
  const OW = w + 2 * padding - filterW / stride + 1;
  const img = input.reshape(OH * OW, filterH, filterW) as nj.NdArray<
    number[][]
  >;
  const imgArr = img.tolist();
  let imgArrIdx = 0;
  const col = (
    nj.zeros([h, w]).reshape(n, d, h, w) as nj.NdArray<number[][][]>
  ).tolist();
  for (let i = 0; i < OH; i = i + stride) {
    for (let j = 0; j < OW; j = j + stride) {
      const imgItem = imgArr[imgArrIdx];
      for (let k = 0; k < filterH; k++) {
        for (let l = 0; l < filterW; l++) {
          col[0][0][i + k][j + l] += imgItem[k][l];
        }
      }
      imgArrIdx++;
    }
  }
  return nj.array(col);
};
