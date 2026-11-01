# Third-party notices

Original GrepMem code is licensed under the MIT license in [LICENSE](LICENSE).
The following adapted portions retain their upstream licenses; the project-level MIT
license does not replace them.

## ContextBench

[ContextBench](https://github.com/RulinShao/ContextBench), revision
`fb533c18c78a82d3f1b2047a9c1700735cf829ae`, is licensed under Apache-2.0.
`agents/contextbench.py` adapts its runner to asynchronous model calls and adds the five
context management strategies, memory handling, and training integration.
The upstream protocol and graders remain dependencies.
The full [Apache-2.0 license](licenses/Apache-2.0.txt) is included.

## rLLM

[rLLM](https://github.com/rllm-org/rllm), revision
`4e828c81680e5a85b638ceedeaf09d1e90109ca8`, is licensed under Apache-2.0.
`patches/rllm-gateway-renderers.patch` contains upstream context and a modification
for renderer API compatibility, also subject to the included Apache-2.0 license.

## AMA-Bench

`ama_eval.py` adapts the judge prompt from
[AMA-Bench](https://github.com/AMA-bench/AMA-bench)'s `utils/evaluation_metrics.py`.
The model-specific thinking suffix is omitted; our file/stream harness integration is separate.
The upstream MIT notice follows:

MIT License

Copyright (c) 2026 AMA-Bench Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Models and datasets

Qwen models, OpenResearcher, BrowseComp-Plus, and AMA-Bench data are downloaded from
their publishers and are not redistributed here. Their terms are independent of this repository's
code license. In particular, do not redistribute decrypted BrowseComp-Plus questions or answers.
