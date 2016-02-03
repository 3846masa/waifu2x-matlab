data = loadjson('scale2.0x_model.json', 'SimplifyCell', 1);
RGB = imread('miku_small.png');
YCbCr = rgb2ycbcr(RGB);

YCbCr_x2 = double(imresize(YCbCr, 2.0, 'nearest'));

ln = length(data);
planes = padarray(YCbCr_x2(:, :, 1), [ln ln], 'replicate') / 255.0;
[rows, cols] = size(planes);

for step = data
  nOut = step.nOutputPlane;
  nIn = step.nInputPlane;
  
  rows = rows - 2;
  cols = cols - 2;
  o_planes = zeros(rows, cols, nOut);

  for i = 1:nOut
    partial = zeros(rows, cols);
    
    for j = 1:nIn
      kernel = zeros(step.kW, step.kH);

      for k = 1:step.kW
        idx = ((i - 1) * step.nInputPlane + j - 1) * 3 + k;
        kernel(k, :) = step.weight(idx, :);
      end
      
      p = conv2(planes(:, :, j), kernel, 'valid');
      partial = partial + p;
    end

    partial = partial + double(step.bias(i));
    partial = max(partial, 0) + min(partial, 0) * 0.1;

    o_planes(:, :, i) = partial;
  end

  planes = o_planes;
end

Y = min(max(planes, 0), 1) * 255.0;

YCbCr_New = YCbCr_x2;
YCbCr_New(:, :, 1) = Y;
YCbCr_New = uint8(YCbCr_New);
RGB_New = ycbcr2rgb(YCbCr_New);

imshow(RGB_New);
imwrite(RGB_New, 'miku_waifu2x.png');
