## COMFYUI-IMAGE-BLENDER

## About
```ComfyuiImageBlender``` is a custom node for ComfyUI.  
You can use it to blend two images together using various modes.  
Currently, 88 blending modes are supported and 45 more are planned to be added.  
Modes logic were borrowed from / inspired by [Krita](https://github.com/KDE/krita) blending modes.

Features:
- 88 blending modes
- support strength parameter
- support mask parameter

<details>
  <summary>Supported blending modes:</summary>
  <ul>
    <li>Arithmetic group
      <ul>
        <li>addition</li>
        <li>divide</li>
        <li>inverse subtract</li>
        <li>multiply</li>
        <li>subtract</li>
      </ul>
    </li>
    <li>Binary group
      <ul>
        <li>AND</li>
        <li>CONVERSE</li>
        <li>IMPLICATION</li>
        <li>NAND</li>
        <li>NOR</li>
        <li>NOT CONVERSE</li>
        <li>NOT IMPLICATION</li>
        <li>OR</li>
        <li>XNOR</li>
        <li>XOR</li>
      </ul>
    </li>
    <li>Darken group
      <ul>
        <li>burn</li>
        <li>darken</li>
        <li>darker color</li>
        <li>easy burn</li>
        <li>fog darken</li>
        <li>gamma dark</li>
        <li>linear burn</li>
        <li>shade</li>
      </ul>
    </li>
    <li>HSI group
      <ul>
        <li>color hsi</li>
        <li>hue hsi</li>
        <li>saturation hsi</li>
        <li>intensity</li>
        <li>decrease saturation hsi</li>
        <li>increase saturation hsi</li>
        <li>decrease intensity</li>
        <li>increase intensity</li>
      </ul>
    </li>
    <li>HSL group
      <ul>
        <li>color hsl</li>
        <li>hue hsl</li>
        <li>saturation hsl</li>
        <li>lightness</li>
        <li>decrease saturation hsl</li>
        <li>increase saturation hsl</li>
        <li>decrease lightness</li>
        <li>increase lightness</li>
      </ul>
    </li>
    <li>HSV group
      <ul>
        <li>color hsv</li>
        <li>hue hsv</li>
        <li>saturation hsv</li>
        <li>value</li>
        <li>decrease saturation hsv</li>
        <li>increase saturation hsv</li>
        <li>decrease value</li>
        <li>increase value</li>
      </ul>
    </li>
    <li>HSY group
      <ul>
        <li>color</li>
        <li>hue</li>
        <li>saturation</li>
        <li>luminosity</li>
        <li>decrease saturation</li>
        <li>increase saturation</li>
        <li>decrease luminosity</li>
        <li>increase luminosity</li>
      </ul>
    </li>
    <li>Lighten group
      <ul>
        <li>color dodge</li>
        <li>linear dodge</li>
        <li>lighten</li>
        <li>linear light</li>
        <li>screen</li>
        <li>pin light</li>
        <li>vivid light</li>
        <li>flat light</li>
        <li>hard light</li>
        <li>soft light (ifs illusions)</li>
        <li>soft light (pegtop-delphi)</li>
        <li>soft light (ps)</li>
        <li>soft light (svg)</li>
        <li>gamma light</li>
        <li>gamma illumination</li>
        <li>lighter color</li>
        <li>p-norm a</li>
        <li>p-norm b</li>
        <li>super light</li>
        <li>tint (ifs illusions)</li>
        <li>fog lighten (ifs illusions)</li>
        <li>easy dodge</li>
        <li>luminosity/shine (sai)</li>
      </ul>
    </li>
    <li>Mix group
      <ul>
        <li>normal</li>
        <li>overlay</li>
      </ul>
    </li>
    <li>Modulo group
      <ul>
        <li>modulo</li>
        <li>divisive modulo</li>
      </ul>
    </li>
    <li>Negative group
      <ul>
        <li>difference</li>
        <li>equivalence</li>
        <li>additive subtractive</li>
        <li>exclusion</li>
        <li>arcus tangent</li>
        <li>negation</li>
      </ul>
    </li>
  </ul>
</details>


## Examples:
<details>
  <summary>Addition</summary>
  <img src="https://github.com/user-attachments/assets/600cd544-840a-49d1-98e6-c69801da31f2" alt="addition">
</details>

<details>
  <summary>Darken</summary>
  <img src="https://github.com/user-attachments/assets/f27aecc3-62dd-463b-8a2f-d3e7d75ac8ce" alt="darken">
</details>

<details>
  <summary>Saturation HSV</summary>
  <img src="https://github.com/user-attachments/assets/1aae01b3-426e-4898-8092-74cc7858914d" alt="saturation_hsv">
</details>


## Comfyui workflow
Feel free to check the example workflow [here](https://github.com/vault-developer/comfyui-image-blender/blob/master/workfow.example.json).  

https://github.com/user-attachments/assets/4b503e6a-cdff-4a3d-ac2b-a482ab0d7d8c


## Installation
You need [comfyui](https://github.com/comfyanonymous/ComfyUI) installed first.  
Then several options are available:
1. You can download or git clone this repository inside ComfyUI/custom_nodes/ directory.  
2. If you use comfy-cli, node can be also downloaded from [comfy registry](https://registry.comfy.org/publishers/vault-developer/nodes/comfyui-image-blender):  
```comfy node registry-install comfyui-image-blender```
3. Comfy-ui manager support will be added when [this pull request](https://github.com/ltdrdata/ComfyUI-Manager/pull/925) is merged.

## Contribution and troubleshooting
This is rough implementation, I will appreciate any feedback.  
Feel free to raise issue if you spot any mistake or have a suggestion.

If you want to contribute, feel free to fork this repository and create a pull request.  
There are still 45 blending modes to be added, so you can help with that.


## Future plans
There are still some things to be done:
- [ ] clean up the code
- [ ] add more blending modes
- [ ] test with PNG images
- [ ] enhance error handling
- [ ] add comfyui manager support
