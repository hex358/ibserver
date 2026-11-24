import { NodeIO } from "@gltf-transform/core";
import { starlinkV2Mini } from "./generators/starlink.ts";
const doc = starlinkV2Mini();
await new NodeIO().write(process.argv[2], doc);
console.log("written", process.argv[2], doc.getRoot().listNodes().length, "nodes");
