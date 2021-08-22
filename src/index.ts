// import { loadMnist } from './data/load-mnist';
import fs from 'fs';
import nj from 'numjs';

// loadMnist();
fs.readFile('train-images-idx3-ubyte', (err, data) => {
  if (err) throw err;
  console.log(data.slice(4, 8)); //
  const offset = 16;
  console.log((data.length - offset) / 784);
  const imgSize = [28, 28];
  // console.log(Array.from(data.slice(offset, offset + 784)));
  console.log(
    nj
      .array(Array.from(data.slice(offset, offset + 784)))
      .reshape(...imgSize)
      .get(14, 14)
    // .get(0)
  );
});
