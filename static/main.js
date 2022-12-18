// const {HandwritingCanvas} = require("handwriting-canvas")

const canvasElement = document.getElementById("draw-area");
const canvas = new HandwritingCanvas(canvasElement);

const clearButtonElement = document.getElementById("clear-button");
clearButtonElement.addEventListener("click", () => {
  canvas.clear();
});

async function preprocess(blob) {
  // 画像を28*28のサイズに変換する
  const canvas = document.createElement("canvas");

  const ctx = canvas.getContext("2d");
  canvas.height = 28;
  canvas.width = 28;

  const bitmap = await createImageBitmap(blob, {
    resizeHeight: 28,
    resizeWidth: 28,
  });
  // キャンバスにリサイズしたbitmapを描く
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, 28, 28);

  // RGBAのalphaの要素だけ取得する
  const alphas = [];
  for (let i = 0; i < imageData.data.length; i++) {
    if (i % 4 === 3) {
      const alpha = imageData.data[i];
      alphas.push(alpha);
    }
  }
  return alphas;
}

async function predict(input) {
  // モデルを読み込む
  const session = await ort.InferenceSession.create("static/model.onxx");

  // 入力データを準備(float形式にする)
  const feeds = {
    float_input: new ort.Tensor("float32", input, [1, 28 * 28])
  };

  //推論を実行する
  const results = await session.run(feeds)

  // 推論結果から必要な要素を取り出す
  return results.probabilities.data
}

const predictButtonElement = document.getElementById("predict-button");
predictButtonElement.addEventListener("click", async () => {
  if (canvas.isEmpty) {
    return;
  }

  //推論を実行する

  const blob = await canvas.toBlob("image/png");
  const input = await preprocess(blob);
  const probabilities = await predict(input);

  // 推論結果の画像を表示
  const imageURL = URL.createObjectURL(blob);
  const imageElement = document.createElement("img");
  imageElement.src = imageURL;

  const resultImageElement = document.getElementById("result-image");

  if (resultImageElement.firstChild) {
    resultImageElement.removeChild(resultImageElement.firstChild);
  }
  resultImageElement.appendChild(imageElement);

  canvas.clear();

  // 推論結果をtbodyに表示
  const tablebodyElement = document.getElementById("result-table-body");

  while (tablebodyElement.firstChild) {
    tablebodyElement.removeChild(tablebodyElement.firstChild);
  }

  let count = 0;
  probabilities.forEach((element) => {
    const tr = document.createElement("tr");

    // 数字
    const tdNumber = document.createElement("td");
    tdNumber.textContent = count;
    tr.appendChild(tdNumber);

    // 確率
    const tdProbability = document.createElement("td");
    tdProbability.textContent = (element * 100).toFixed(1);
    tr.appendChild(tdProbability);

    tablebodyElement.appendChild(tr);
    count++;
  });
});
