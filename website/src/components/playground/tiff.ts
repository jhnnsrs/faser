/**
 * Minimal writer for a multi-page 32-bit float TIFF, one page per z-slice,
 * with an ImageJ description and resolution tags so Fiji / napari pick up
 * the voxel size. Little-endian, uncompressed.
 */

const TYPE_ASCII = 2;
const TYPE_SHORT = 3;
const TYPE_LONG = 4;
const TYPE_RATIONAL = 5;

interface Entry {
  tag: number;
  type: number;
  count: number;
  /** Inline value (<= 4 bytes) or the offset of an out-of-line payload. */
  value: number;
}

export function writeTiff(
  data: Float32Array,
  nz: number,
  ny: number,
  nx: number,
  spacing: { dx: number; dy: number; dz: number; unit?: string },
): Blob {
  const unit = spacing.unit ?? 'micron';
  const description = `ImageJ=1.54f\nimages=${nz}\nslices=${nz}\nunit=${unit}\nspacing=${spacing.dz}\nloop=false\n`;
  const descBytes = new TextEncoder().encode(description + '\0');
  const sliceBytes = nx * ny * 4;

  // Rational resolution: pixels per unit = 1 / dx, encoded as num/den.
  const rational = (perUnit: number) => {
    const den = 1_000_000;
    const num = Math.round(perUnit * den);
    const b = new Uint8Array(8);
    new DataView(b.buffer).setUint32(0, num, true);
    new DataView(b.buffer).setUint32(4, den, true);
    return b;
  };
  const xres = rational(1 / spacing.dx);
  const yres = rational(1 / spacing.dy);

  // Layout: header (8) | per page: IFD + out-of-line values | all pixel data.
  const ENTRIES = 15;
  const ifdSize = 2 + ENTRIES * 12 + 4;
  const raw = ifdSize + descBytes.length + 16; // + description + xres + yres
  const pageMetaSize = Math.ceil(raw / 4) * 4; // keep the pixel block 4-byte aligned
  const headerSize = 8;
  const pixelsOffset = headerSize + nz * pageMetaSize;
  const total = pixelsOffset + nz * sliceBytes;
  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  // Header: "II", 42, offset of the first IFD
  view.setUint8(0, 0x49);
  view.setUint8(1, 0x49);
  view.setUint16(2, 42, true);
  view.setUint32(4, headerSize, true);

  for (let z = 0; z < nz; z++) {
    const ifdOffset = headerSize + z * pageMetaSize;
    const descOffset = ifdOffset + ifdSize;
    const xresOffset = descOffset + descBytes.length;
    const yresOffset = xresOffset + 8;
    bytes.set(descBytes, descOffset);
    bytes.set(xres, xresOffset);
    bytes.set(yres, yresOffset);

    // Sorted by tag, as the spec requires.
    const entries: Entry[] = [
      { tag: 254, type: TYPE_LONG, count: 1, value: 0 }, // NewSubfileType
      { tag: 256, type: TYPE_LONG, count: 1, value: nx }, // ImageWidth
      { tag: 257, type: TYPE_LONG, count: 1, value: ny }, // ImageLength
      { tag: 258, type: TYPE_SHORT, count: 1, value: 32 }, // BitsPerSample
      { tag: 259, type: TYPE_SHORT, count: 1, value: 1 }, // Compression: none
      { tag: 262, type: TYPE_SHORT, count: 1, value: 1 }, // Photometric: BlackIsZero
      { tag: 270, type: TYPE_ASCII, count: descBytes.length, value: descOffset }, // ImageDescription
      { tag: 273, type: TYPE_LONG, count: 1, value: pixelsOffset + z * sliceBytes }, // StripOffsets
      { tag: 277, type: TYPE_SHORT, count: 1, value: 1 }, // SamplesPerPixel
      { tag: 278, type: TYPE_LONG, count: 1, value: ny }, // RowsPerStrip
      { tag: 279, type: TYPE_LONG, count: 1, value: sliceBytes }, // StripByteCounts
      { tag: 282, type: TYPE_RATIONAL, count: 1, value: xresOffset }, // XResolution
      { tag: 283, type: TYPE_RATIONAL, count: 1, value: yresOffset }, // YResolution
      { tag: 296, type: TYPE_SHORT, count: 1, value: 1 }, // ResolutionUnit: none (ImageJ uses `unit=`)
      { tag: 339, type: TYPE_SHORT, count: 1, value: 3 }, // SampleFormat: IEEE float
    ];
    if (entries.length !== ENTRIES) throw new Error('tiff: entry count mismatch');

    let off = ifdOffset;
    view.setUint16(off, entries.length, true);
    off += 2;
    for (const e of entries) {
      view.setUint16(off, e.tag, true);
      view.setUint16(off + 2, e.type, true);
      view.setUint32(off + 4, e.count, true);
      if (e.type === TYPE_SHORT && e.count === 1) view.setUint16(off + 8, e.value, true);
      else view.setUint32(off + 8, e.value, true);
      off += 12;
    }
    const next = z + 1 < nz ? headerSize + (z + 1) * pageMetaSize : 0;
    view.setUint32(off, next, true);
  }

  // Pixel data (little-endian float32; the platform is little-endian in practice)
  new Float32Array(buf, pixelsOffset, nz * ny * nx).set(data);

  return new Blob([buf], { type: 'image/tiff' });
}
