import nj from 'numjs';

const a = nj.array([
  [
    [1, 2, 4, 5, 10, 11, 13, 14],
    [2, 3, 5, 6, 11, 12, 14, 15],
    [4, 5, 7, 8, 13, 14, 16, 17],
    [5, 6, 8, 9, 14, 15, 17, 18],
  ],
]);

// console.log(a.slice(0, [0, 4]).tolist());
// console.log(a.slice(0, [4, 8]).tolist());
// console.log(
//   nj.array(
//     a.tolist().map((convoluteOutput) => {
//       return Math.max(...convoluteOutput);
//     })
//   )
// );

console.log(a.multiply(nj.zeros([1, 4, 8]).reshape(1, 4, 8)));
