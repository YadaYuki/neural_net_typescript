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
