# Changelog

## 1.0.0 (2026-10-25)


### Features

* configure Release Please automation ([#67](https://github.com/atankeev/kmeans/issues/67)) ([cbe5e19](https://github.com/atankeev/kmeans/commit/cbe5e19103fd42e6950d55f508cdcbcaec0037d8))
* simplify batch clustering API ([#30](https://github.com/atankeev/kmeans/issues/30)) ([3d7de49](https://github.com/atankeev/kmeans/commit/3d7de496790caddfd976a9f7599a3ad80e7ddc54))


### Bug Fixes

* align convergence tolerance with sklearn ([afb8460](https://github.com/atankeev/kmeans/commit/afb846009b07d80b55e557324f54d648f6c8e584))
* compare previous centers before updating in lloyd ([0a62169](https://github.com/atankeev/kmeans/commit/0a621691f4aaa92bb5ef5b884952edd040f80bd4))
* expose clustering convergence failure ([aa5ed11](https://github.com/atankeev/kmeans/commit/aa5ed1161c64088c27647e51243f4416524accb2))
* handle empty k-means clusters deterministically ([f32d162](https://github.com/atankeev/kmeans/commit/f32d162a5f1ed99f90072d7bec9274210632ea41))
* isolate centroid row capacity ([b881108](https://github.com/atankeev/kmeans/commit/b881108aa4963d6b714572e2be3cee23324b5876))
* make Kmeans calls concurrency-safe ([#8](https://github.com/atankeev/kmeans/issues/8)) ([3c1f5df](https://github.com/atankeev/kmeans/commit/3c1f5df392281f07601093c82c4772fa1686dfea))
* preserve internal inertia ordering ([#13](https://github.com/atankeev/kmeans/issues/13)) ([9c9eacc](https://github.com/atankeev/kmeans/commit/9c9eacc9dc44ca36ea8e8ad41a5c952620f9032c))
* prevent numerical overflow for finite input ([#9](https://github.com/atankeev/kmeans/issues/9)) ([0fc0768](https://github.com/atankeev/kmeans/commit/0fc07688807e96e1f12cf3248790f179b66b6a00))
* reject datasets with zero feature dimensions ([a5602fe](https://github.com/atankeev/kmeans/commit/a5602fe5bb11fd52089f741e49b94657f4b51c7a))
* reject non-finite input values ([4463452](https://github.com/atankeev/kmeans/commit/44634528923fc36225aa4a408ceb0ab715d75968))
* report invalid functional option values ([44a4830](https://github.com/atankeev/kmeans/commit/44a4830ab045849be0298a0eef81aafcea5b00a7))
* return ErrInvalidK for excessive cluster counts ([#12](https://github.com/atankeev/kmeans/issues/12)) ([dff728d](https://github.com/atankeev/kmeans/commit/dff728d45d2eadaf6cba1d006eea9c24a25bd2bc))
* validate centroid initialization method ([fd199d6](https://github.com/atankeev/kmeans/commit/fd199d61697ae43b5c64e4b2ad9211d2f31f8443))
