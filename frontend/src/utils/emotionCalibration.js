export const CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

export const EMOJI_MAP = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprise: "😲",
};

export function topClassFromMap(probabilities) {
  let bestClass = CLASSES[0];
  let bestValue = -Infinity;

  for (const cls of CLASSES) {
    const value = Number(probabilities?.[cls] ?? 0);
    if (value > bestValue) {
      bestValue = value;
      bestClass = cls;
    }
  }

  return bestClass;
}

export function roundMap(probabilities) {
  const out = {};
  for (const cls of CLASSES) {
    out[cls] = Number(Number(probabilities?.[cls] ?? 0).toFixed(1));
  }
  return out;
}

export function applyCalibration(probabilities) {
  const adjusted = { ...probabilities };

  adjusted.neutral = Number(adjusted.neutral ?? 0) + 1.8;
  adjusted.happy = Number(adjusted.happy ?? 0) + 4.0;
  adjusted.angry = Number(adjusted.angry ?? 0) + 6.0;
  adjusted.disgust = Math.max(0, Number(adjusted.disgust ?? 0) - 4.0);
  adjusted.sad = Number(adjusted.sad ?? 0) + 2.2;

  if (
    Number(adjusted.happy ?? 0) < 52 &&
    Number(adjusted.angry ?? 0) < 42 &&
    Math.abs(Number(adjusted.sad ?? 0) - Number(adjusted.neutral ?? 0)) <= 8
  ) {
    adjusted.sad += 6.0;
    adjusted.neutral = Math.max(0, Number(adjusted.neutral ?? 0) - 2.5);
  }

  if (
    Number(adjusted.sad ?? 0) >= 14 &&
    Number(adjusted.neutral ?? 0) > Number(adjusted.sad ?? 0) &&
    Number(adjusted.neutral ?? 0) - Number(adjusted.sad ?? 0) <= 10
  ) {
    adjusted.sad += 5.0;
    adjusted.neutral = Math.max(0, Number(adjusted.neutral ?? 0) - 2.0);
  }

  const neutralNow = Number(adjusted.neutral ?? 0);
  const neutralToSad = Math.min(neutralNow * 0.6, 20.0);
  adjusted.neutral = Math.max(0, neutralNow - neutralToSad);
  adjusted.sad = Number(adjusted.sad ?? 0) + neutralToSad;

  if (
    Number(adjusted.happy ?? 0) >= 16 &&
    Number(adjusted.neutral ?? 0) >= 18 &&
    Math.abs(Number(adjusted.happy ?? 0) - Number(adjusted.neutral ?? 0)) <= 10
  ) {
    adjusted.happy += 4.0;
  }

  if (
    Number(adjusted.angry ?? 0) >= 18 &&
    Math.abs(Number(adjusted.angry ?? 0) - Number(adjusted.neutral ?? 0)) <= 6
  ) {
    adjusted.angry += 3.5;
  }

  if (
    Number(adjusted.angry ?? 0) >= 11 &&
    Number(adjusted.disgust ?? 0) >= Number(adjusted.angry ?? 0) &&
    Number(adjusted.disgust ?? 0) - Number(adjusted.angry ?? 0) <= 10
  ) {
    adjusted.angry += 6.0;
    adjusted.disgust = Math.max(0, Number(adjusted.disgust ?? 0) - 3.0);
  }

  const disgustNow = Number(adjusted.disgust ?? 0);
  const transfer = Math.min(disgustNow * 0.25, 8.0);
  adjusted.disgust = Math.max(0, disgustNow - transfer);
  adjusted.angry = Number(adjusted.angry ?? 0) + transfer;

  return adjusted;
}
