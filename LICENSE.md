# LangOR License and Third-Party Notices

## Scope

LangOR includes original contributions, modified upstream code, and bundled third-party components. It is not offered as a repository-wide MIT or Apache-2.0 project.

To the extent that the LangOR contributors hold the necessary rights, their project-specific contributions and modifications are made available under the non-commercial research and evaluation terms of the Gaussian-Splatting License reproduced below. For this grant, references to the Software and Work include those contributions and modifications; this does not transfer their ownership to the upstream licensors. Commercial use is not granted by this notice.

Third-party components remain subject to their respective licenses. This document does not replace their copyright holders, remove existing attribution, or grant rights that the LangOR contributors do not hold. Preserve this file and all existing third-party license files and copyright notices when redistributing the corresponding code.

The third-party provisions below apply to their identified components, rather than extending one component's license to unrelated files. Data, annotations, embeddings, downloaded model weights, and generated assets are not licensed by this code license; their applicable permissions must be established separately.

## LaGa attribution and unresolved authorization

Parts of the training, scene representation, rendering, and inference implementation are derived from [LaGa](https://github.com/SJTU-DeepVisionLab/LaGa), by Jiazhong Cen and collaborators. Existing `Modified by Jiazhong Cen, 2025` notices are retained. LangOR modifications include pipeline refactoring, configuration and naming changes, output simplification, and support for partially annotated views. These modifications do not extinguish upstream rights.

Reference: Jiazhong Cen et al., *Tackling View-Dependent Semantics in 3D Language Gaussian Splatting*, ICML 2025.

The inspected LaGa root at commit `9df7586d3f5e0a9b3ad4ae1083eef9707f756e70` contains no standalone license file, and its README does not establish a separate project-wide license. This observation does not establish that every file lacks a license. The exact LaGa revision used to create this repository has not been established. Authorization for LaGa-specific material not covered by an applicable file-level or upstream license remains unconfirmed and must be resolved with the relevant rights holders before redistribution. This document does not itself resolve that authorization.

## Gaussian Splatting, rasterization, and Simple KNN

Applies to code carrying the Inria/GRAPHDECO notices in the project root, `arguments/`, `scene/`, `gaussian_renderer/`, and `utils/`, and to inherited code in:

- `submodules/diff-gaussian-rasterization/`
- `submodules/diff-gaussian-rasterization_contrastive_f/`
- `submodules/simple-knn/`

Separately licensed embedded components are identified below. Modified components may also contain contributions whose rights are described in the LaGa section.

License sources checked on 2026-11-10:

- [Gaussian Splatting license](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md)
- [Rasterizer license](https://raw.githubusercontent.com/graphdeco-inria/diff-gaussian-rasterization/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/LICENSE.md)
- [Simple KNN license](https://gitlab.inria.fr/bkerbl/simple-knn/-/raw/main/LICENSE.md)

The cited commits identify the license snapshots, not proven base revisions of the local source. Simple KNN supplied the same text as the Gaussian Splatting snapshot; the rasterizer supplied the same terms without the SSIM attribution appendix. The complete Gaussian Splatting text follows unchanged.

Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
Programmes (APP).  

The *Software* is still being developed by the *Licensor*.  

*Licensor*'s goal is to allow the research community to use, test and evaluate  
the *Software*.  

## 1.  Definitions  

*Licensee* means any person or entity that uses the *Software* and distributes  
its *Work*.  

*Licensor* means the owners of the *Software*, i.e Inria and MPII  

*Software* means the original work of authorship made available under this  
License ie gaussian-splatting.  

*Work* means the *Software* and any additions to or derivative works of the  
*Software* that are made available under this License.  


## 2.  Purpose  
This license is intended to define the rights granted to the *Licensee* by  
Licensors under the *Software*.  

## 3.  Rights granted  

For the above reasons Licensors have decided to distribute the *Software*.  
Licensors grant non-exclusive rights to use the *Software* for research purposes  
to research users (both academic and industrial), free of charge, without right  
to sublicense.. The *Software* may be used "non-commercially", i.e., for research  
and/or evaluation purposes only.  

Subject to the terms and conditions of this License, you are granted a  
non-exclusive, royalty-free, license to reproduce, prepare derivative works of,  
publicly display, publicly perform and distribute its *Work* and any resulting  
derivative works in any form.  

## 4.  Limitations  

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a) you do  
so under this License, (b) you include a complete copy of this License with  
your distribution, and (c) you retain without modification any copyright,  
patent, trademark, or attribution notices that are present in the *Work*.  

**4.2 Derivative Works.** You may specify that additional or different terms apply  
to the use, reproduction, and distribution of your derivative works of the *Work*  
("Your Terms") only if (a) Your Terms provide that the use limitation in  
Section 2 applies to your derivative works, and (b) you identify the specific  
derivative works that are subject to Your Terms. Notwithstanding Your Terms,  
this License (including the redistribution requirements in Section 3.1) will  
continue to apply to the *Work* itself.  

**4.3** Any other use without of prior consent of Licensors is prohibited. Research  
users explicitly acknowledge having received from Licensors all information  
allowing to appreciate the adequacy between of the *Software* and their needs and  
to undertake all necessary precautions for its execution and use.  

**4.4** The *Software* is provided both as a compiled library file and as source  
code. In case of using the *Software* for a publication or other results obtained  
through the use of the *Software*, users are strongly encouraged to cite the  
corresponding publications as explained in the documentation of the *Software*.  

## 5.  Disclaimer  

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL PURPOSES  
WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS. YOU MUST CONTACT INRIA FOR ANY  
UNAUTHORIZED USE: stip-sophia.transfert@inria.fr . ANY SUCH ACTION WILL  
CONSTITUTE A FORGERY. THIS *SOFTWARE* IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES  
OF ANY NATURE AND ANY EXPRESS OR IMPLIED WARRANTIES, WITH REGARDS TO COMMERCIAL  
USE, PROFESSIONNAL USE, LEGAL OR NOT, OR OTHER, OR COMMERCIALISATION OR  
ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN NO EVENT, SHALL INRIA OR THE  
AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR  
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE  
GOODS OR SERVICES, LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION)  
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT  
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING FROM, OUT OF OR  
IN CONNECTION WITH THE *SOFTWARE* OR THE USE OR OTHER DEALINGS IN THE *SOFTWARE*.  

## 6.  Files subject to permissive licenses
The contents of the file ```utils/loss_utils.py``` are based on publicly available code authored by Evan Su, which falls under the permissive MIT license. 

Title: pytorch-ssim\
Project code: https://github.com/Po-Hsun-Su/pytorch-ssim\
Copyright Evan Su, 2017\
License: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt (MIT)

## LPIPS implementation

Applies to `lpipsPyTorch/`, derived from [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch). Its local VGG16 initialization uses the torchvision weights API. The source license is preserved below from [commit 016e0ae](https://raw.githubusercontent.com/S-aiueo32/lpips-pytorch/016e0ae19ceff83450590738948941a7eb9b95b7/LICENSE). Downloaded weights have separate provenance and are not relicensed here.

BSD 2-Clause License

Copyright (c) 2020, Sou Uchida
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## COLMAP Python readers

Applies to COLMAP-derived readers in `scene/colmap_loader.py`. The local file explicitly cites COLMAP reader scripts. The following notice is reproduced from [COLMAP 3.8 read_write_model.py](https://raw.githubusercontent.com/colmap/colmap/3.8/scripts/python/read_write_model.py); the precise original import revision has not been established. Existing Inria notices also remain in place.

Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

## SSIM implementation

The SSIM implementation in `utils/loss_utils.py` derives from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim). The Gaussian Splatting license above identifies Evan Su, 2017, and the MIT license. The standard MIT terms for that component are reproduced here.

Copyright Evan Su, 2017

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Other retained component licenses

The following existing notices form part of the distribution and must remain with their components:

| Component | Local files | Applicable notice |
| --- | --- | --- |
| PlenOctree spherical harmonics | `utils/sh_utils.py` | Full BSD-2-Clause notice in the file header; Copyright 2021 The PlenOctree Authors. |
| PyTorch3D 0.7.2 source | `submodules/pytorch3d-0.7.2/` | [LICENSE](submodules/pytorch3d-0.7.2/LICENSE) and [LICENSE-3RD-PARTY](submodules/pytorch3d-0.7.2/LICENSE-3RD-PARTY). The source remains bundled even though installation uses version 0.7.7. |
| GLM | Both rasterizer directories under `third_party/glm/` | Each existing `copying.txt` contains the offered licenses and copyright notices. |
| stb_image_write | Both rasterizer directories under `third_party/stbi_image_write.h` | The embedded license text at the end of each header remains applicable. |

Packages installed through `requirements.txt`, separately installed PyTorch3D, and downloaded encoders or weights retain their own licenses. Merely listing a dependency does not relicense it under the LangOR terms.
