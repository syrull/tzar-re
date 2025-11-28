#!/usr/bin/env python3
"""
Tzar: The Burden of the Crown - WDT Archive Tool

A complete CLI tool for working with Tzar WDT archives:
- Extract files from WDT archives
- Create/bundle WDT archives from directories
- Convert BMP images to Tzar-compatible format
- Validate assets before bundling

Usage:
    tzar_wdt.py extract <archive.wdt> -o <output_dir>
    tzar_wdt.py bundle <output.wdt> -d <source_dir>
    tzar_wdt.py list <archive.wdt>
    tzar_wdt.py convert-bmp <image.bmp>
    tzar_wdt.py validate -d <directory>

See 'tzar_wdt.py <command> --help' for more information.
"""

import struct
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional


# =============================================================================
# BMP Utilities
# =============================================================================

def get_bmp_info(filepath: Path) -> dict:
    """Read BMP header information."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read(138)
    except Exception as e:
        return {'valid': False, 'error': str(e)}
    
    if len(data) < 30 or data[:2] != b'BM':
        return {'valid': False, 'error': 'Not a BMP file'}
    
    file_size = struct.unpack_from('<I', data, 2)[0]
    data_offset = struct.unpack_from('<I', data, 10)[0]
    dib_size = struct.unpack_from('<I', data, 14)[0]
    width = struct.unpack_from('<i', data, 18)[0]
    height = struct.unpack_from('<i', data, 22)[0]
    bpp = struct.unpack_from('<H', data, 28)[0]
    compression = struct.unpack_from('<I', data, 30)[0] if len(data) > 30 else 0
    
    # Tzar uses:
    # - 16-bit RGB555 for color images (screens, sprites)
    # - 8-bit indexed for masks/collision maps
    # - 24-bit RGB for some screen images
    # All must use BITMAPINFOHEADER (40 bytes) with no compression
    is_compatible = (dib_size == 40 and compression == 0 and bpp in (8, 16, 24))
    
    return {
        'valid': True,
        'file_size': file_size,
        'data_offset': data_offset,
        'dib_size': dib_size,
        'width': width,
        'height': abs(height),
        'height_raw': height,
        'bpp': bpp,
        'compression': compression,
        'is_tzar_compatible': is_compatible,
    }


def convert_bmp_to_rgb555(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Convert any BMP to 16-bit RGB555 format compatible with Tzar.
    
    Returns True if conversion was performed, False if already compatible.
    """
    info = get_bmp_info(input_path)
    if not info['valid']:
        raise ValueError(f"Invalid BMP file: {info.get('error', 'unknown error')}")
    
    if info['is_tzar_compatible']:
        return False
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for BMP conversion: pip install Pillow")
    
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    pixels = img.load()
    
    # Build pixel data (RGB555, bottom-to-top row order)
    pixel_data = bytearray()
    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = pixels[x, y]
            r5 = (r >> 3) & 0x1F
            g5 = (g >> 3) & 0x1F
            b5 = (b >> 3) & 0x1F
            pixel16 = (r5 << 10) | (g5 << 5) | b5
            pixel_data.extend(struct.pack('<H', pixel16))
    
    # Build BMP file
    bmp_data = bytearray()
    file_size = 54 + len(pixel_data)
    
    # File header (14 bytes)
    bmp_data.extend(b'BM')
    bmp_data.extend(struct.pack('<I', file_size))
    bmp_data.extend(struct.pack('<HH', 0, 0))
    bmp_data.extend(struct.pack('<I', 54))
    
    # DIB header - BITMAPINFOHEADER (40 bytes)
    bmp_data.extend(struct.pack('<I', 40))
    bmp_data.extend(struct.pack('<i', width))
    bmp_data.extend(struct.pack('<i', height))
    bmp_data.extend(struct.pack('<H', 1))
    bmp_data.extend(struct.pack('<H', 16))
    bmp_data.extend(struct.pack('<I', 0))
    bmp_data.extend(struct.pack('<I', 0))
    bmp_data.extend(struct.pack('<i', 0))
    bmp_data.extend(struct.pack('<i', 0))
    bmp_data.extend(struct.pack('<I', 0))
    bmp_data.extend(struct.pack('<I', 0))
    
    bmp_data.extend(pixel_data)
    
    out_path = output_path or input_path
    with open(out_path, 'wb') as f:
        f.write(bmp_data)
    
    return True


def validate_bmps_in_directory(directory: Path, fix: bool = False, verbose: bool = True) -> tuple[int, int, int]:
    """
    Validate (and optionally fix) all BMP files in a directory.
    
    Returns: (total_files, ok_files, problematic_files)
    """
    total = 0
    ok = 0
    problems = 0
    
    # Find all BMP files (case-insensitive)
    bmp_files = list(directory.rglob('*.[Bb][Mm][Pp]'))
    
    for bmp_path in sorted(bmp_files):
        total += 1
        info = get_bmp_info(bmp_path)
        
        if not info['valid']:
            if verbose:
                print(f"⚠  {bmp_path.relative_to(directory)}: Invalid ({info.get('error', 'unknown')})")
            problems += 1
            continue
        
        if info['is_tzar_compatible']:
            ok += 1
            if verbose:
                bpp_type = {8: "indexed", 16: "RGB555", 24: "RGB"}.get(info['bpp'], str(info['bpp']))
                print(f"✓  {bmp_path.relative_to(directory)}: OK ({info['bpp']}-bit {bpp_type})")
        else:
            if fix:
                try:
                    convert_bmp_to_rgb555(bmp_path)
                    problems += 1  # Count as changed
                    if verbose:
                        print(f"✔  {bmp_path.relative_to(directory)}: Converted ({info['bpp']}-bit → 16-bit)")
                except Exception as e:
                    problems += 1
                    if verbose:
                        print(f"❌ {bmp_path.relative_to(directory)}: Error - {e}")
            else:
                problems += 1
                if verbose:
                    issues = []
                    if info['bpp'] not in (8, 16, 24):
                        issues.append(f"bpp={info['bpp']} (need 8, 16, or 24)")
                    if info['dib_size'] != 40:
                        issues.append(f"header={info['dib_size']}B (need 40)")
                    if info['compression'] != 0:
                        issues.append(f"compression={info['compression']} (need 0)")
                    print(f"✗  {bmp_path.relative_to(directory)}: INCOMPATIBLE ({', '.join(issues)})")
    
    return total, ok, problems


# =============================================================================
# LZSS Decompressor
# =============================================================================

class LZSSDecompressor:
    """LZSS Decompressor for Tzar WDT files (mode 0xC4)."""
    
    MAGIC = b'LZSS'
    WINDOW_SIZE = 0x1000
    
    def decompress_file(self, filepath: Path) -> bytes:
        """Decompress an entire WDT file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return self.decompress(data)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress WDT data."""
        if data[:4] != self.MAGIC:
            raise ValueError(f"Invalid magic: expected {self.MAGIC!r}, got {data[:4]!r}")
        
        total_size = struct.unpack_from('<I', data, 4)[0]
        chunk_size = struct.unpack_from('<I', data, 8)[0]
        reserved = struct.unpack_from('<H', data, 12)[0]
        mode = reserved & 0xFF
        
        if mode != 0xC4:
            raise ValueError(f"Unsupported LZSS mode: 0x{mode:02x}")
        
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        
        chunk_offsets = []
        for i in range(num_chunks):
            chunk_offsets.append(struct.unpack_from('<I', data, 14 + i * 4)[0])
        chunk_offsets.append(len(data))
        
        output = bytearray()
        for i in range(num_chunks):
            dec_size = chunk_size if i < num_chunks - 1 else total_size - (chunk_size * (num_chunks - 1))
            chunk_data = data[chunk_offsets[i]:chunk_offsets[i + 1]]
            chunk_output = self._decode_chunk(chunk_data, dec_size)
            output.extend(chunk_output)
        
        return bytes(output[:total_size])
    
    def _decode_chunk(self, src: bytes, max_output: int) -> bytes:
        """Decode a single LZSS chunk."""
        output = bytearray()
        src_ptr = 0
        bit_state = 0
        
        def read_u32_at(pos: int) -> int:
            if pos >= len(src):
                return 0
            chunk = src[pos:pos + 4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            return struct.unpack('<I', chunk)[0]
        
        def bswap32(val: int) -> int:
            return (((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
                    ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000))
        
        src_end = len(src) - 1
        
        while len(output) < max_output and src_ptr < src_end:
            raw = read_u32_at(src_ptr)
            swapped = bswap32(raw)
            bit_pos = bit_state & 0x07
            shifted = (swapped << bit_pos) & 0xFFFFFFFF
            iVar2 = shifted
            uVar3 = (iVar2 << 1) & 0xFFFFFFFF
            
            if iVar2 & 0x80000000:
                advance = (1 if bit_state > 0xdffe else 0) + 1
                src_ptr += advance
                bit_state = (bit_state + 0x2001) & 0xff07
                literal = (uVar3 >> 24) & 0xFF
                output.append(literal)
            else:
                advance = (1 if bit_state > 0xdffe else 0) + 2
                src_ptr += advance
                bit_state = (bit_state + 0x2001) & 0xff07
                offset = (uVar3 >> 20) & 0xFFF
                length_bits = (uVar3 >> 16) & 0x0F
                
                if offset == 0:
                    skip_len = length_bits + 2
                    output.extend(b'\x00' * skip_len)
                else:
                    back_dist = self.WINDOW_SIZE - offset
                    if length_bits & 1:
                        if len(output) >= back_dist:
                            output.append(output[-back_dist])
                        else:
                            output.append(0)
                    pair_count = (length_bits >> 1) + 1
                    for _ in range(pair_count):
                        if len(output) >= back_dist:
                            output.append(output[-back_dist])
                        else:
                            output.append(0)
                        if len(output) >= back_dist:
                            output.append(output[-back_dist])
                        else:
                            output.append(0)
        
        return bytes(output[:max_output])


# =============================================================================
# HMMSYS PackFile Parser/Creator
# =============================================================================

@dataclass
class HMMSYSFileEntry:
    """Represents a file entry in the HMMSYS PackFile."""
    name: str
    offset: int
    size: int
    directory: str
    raw_stored: str = ""
    b0: int = 0
    b1: int = 0


@dataclass
class FileEntry:
    """Represents a file to be packed."""
    path: Path
    archive_name: str
    size: int


class HMMSYSUnpacker:
    """Unpacker for HMMSYS PackFile format."""
    
    MAGIC = b'HMMSYS PackFile\n'
    HEADER_SIZE = 0x28
    
    def __init__(self, data: bytes, debug: bool = False):
        self.data = data
        self.debug = debug
        self.file_count = 0
        self.entry_table_size = 0
        self.entries: list[HMMSYSFileEntry] = []
        self._prev_full_path = ""
        
        self._parse_header()
        self._parse_entries()
    
    def _parse_header(self):
        if len(self.data) < self.HEADER_SIZE:
            raise ValueError("Data too short for HMMSYS header")
        
        magic = self.data[0:16]
        if magic != self.MAGIC:
            raise ValueError(f"Invalid HMMSYS magic: {magic!r}")
        
        self._eof_marker = struct.unpack_from('<I', self.data, 0x10)[0]
        self.file_count = struct.unpack_from('<I', self.data, 0x20)[0]
        self.entry_table_size = struct.unpack_from('<I', self.data, 0x24)[0]
        
        if self.debug:
            print(f"HMMSYS: files={self.file_count}, table_size={self.entry_table_size}")
    
    def _parse_entries(self):
        pos = self.HEADER_SIZE
        entry_end = self.HEADER_SIZE + self.entry_table_size
        
        for entry_idx in range(self.file_count):
            if pos >= entry_end or pos >= len(self.data) - 10:
                break
            
            entry, consumed = self._parse_entry(pos, entry_idx)
            if entry:
                self.entries.append(entry)
                if self.debug:
                    print(f"  [{entry_idx:3d}] {entry.name}")
            
            if consumed == 0:
                entry, consumed = self._parse_entry_bruteforce(pos, entry_idx)
                if entry:
                    self.entries.append(entry)
                else:
                    break
            
            pos += consumed
    
    def _parse_entry(self, pos: int, entry_idx: int) -> tuple[HMMSYSFileEntry | None, int]:
        if entry_idx == 0:
            name_len = struct.unpack_from('<H', self.data, pos)[0]
            full_path = self.data[pos + 2:pos + 2 + name_len].decode('latin-1')
            offset = struct.unpack_from('<I', self.data, pos + 2 + name_len)[0]
            size = struct.unpack_from('<I', self.data, pos + 2 + name_len + 4)[0]
            
            self._prev_full_path = full_path
            directory = full_path.rsplit('\\', 1)[0] if '\\' in full_path else ""
            
            return HMMSYSFileEntry(
                name=full_path, offset=offset, size=size,
                directory=directory, raw_stored=full_path, b0=0, b1=0
            ), 2 + name_len + 8
        
        b0 = self.data[pos]
        b1 = self.data[pos + 1]
        stored_len = b0 - b1
        
        if stored_len <= 0 or stored_len > 100:
            return None, 0
        
        stored = self.data[pos + 2:pos + 2 + stored_len].decode('latin-1', errors='replace')
        offset = struct.unpack_from('<I', self.data, pos + 2 + stored_len)[0]
        size = struct.unpack_from('<I', self.data, pos + 2 + stored_len + 4)[0]
        
        if not (0x50 < offset < len(self.data) and 0 < size < 50_000_000):
            return None, 0
        
        full_name = self._prev_full_path[:b1] + stored
        self._prev_full_path = full_name
        directory = full_name.rsplit('\\', 1)[0] if '\\' in full_name else ""
        
        return HMMSYSFileEntry(
            name=full_name, offset=offset, size=size,
            directory=directory, raw_stored=stored, b0=b0, b1=b1
        ), 2 + stored_len + 8
    
    def _parse_entry_bruteforce(self, pos: int, entry_idx: int) -> tuple[HMMSYSFileEntry | None, int]:
        for try_len in range(3, min(60, len(self.data) - pos - 10)):
            name_bytes = self.data[pos + 2:pos + 2 + try_len]
            if b'\x00' in name_bytes:
                continue
            try:
                name_str = name_bytes.decode('latin-1')
                if not all(c.isalnum() or c in '._-\\/ ' for c in name_str):
                    continue
            except:
                continue
            
            offset = struct.unpack_from('<I', self.data, pos + 2 + try_len)[0]
            size = struct.unpack_from('<I', self.data, pos + 2 + try_len + 4)[0]
            
            if 0x50 < offset < len(self.data) and 0 < size < 50_000_000:
                b0 = self.data[pos]
                b1 = self.data[pos + 1]
                full_name = name_str
                self._prev_full_path = full_name
                directory = full_name.rsplit('\\', 1)[0] if '\\' in full_name else ""
                
                return HMMSYSFileEntry(
                    name=full_name, offset=offset, size=size,
                    directory=directory, raw_stored=name_str, b0=b0, b1=b1
                ), 2 + try_len + 8
        
        return None, 0
    
    def extract_file(self, entry: HMMSYSFileEntry) -> bytes:
        return self.data[entry.offset:entry.offset + entry.size]
    
    def extract_all(self, output_dir: Path, use_raw_names: bool = False, verbose: bool = True):
        output_dir = Path(output_dir)
        
        for entry in self.entries:
            rel_path = entry.name.replace('\\', '/')
            file_path = output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = self.extract_file(entry)
            file_path.write_bytes(content)
            if verbose:
                print(f"Extracted: {rel_path} ({entry.size} bytes)")
    
    def list_files(self) -> Iterator[HMMSYSFileEntry]:
        yield from self.entries


class HMMSYSPacker:
    """Packer for HMMSYS PackFile format."""
    
    MAGIC = b'HMMSYS PackFile\n'
    HEADER_SIZE = 0x28
    
    def __init__(self):
        self.entries: list[FileEntry] = []
    
    def add_file(self, disk_path: Path, archive_name: str):
        size = disk_path.stat().st_size
        self.entries.append(FileEntry(path=disk_path, archive_name=archive_name, size=size))
    
    def add_directory(self, root_dir: Path, prefix: str = ""):
        root_dir = Path(root_dir)
        for item in sorted(root_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(root_dir)
                archive_name = str(rel_path).replace('/', '\\')
                if prefix:
                    archive_name = prefix + '\\' + archive_name
                self.add_file(item, archive_name)
    
    def _compute_prefix_length(self, prev: str, current: str) -> int:
        min_len = min(len(prev), len(current))
        prefix_len = 0
        for i in range(min_len):
            if prev[i] == current[i]:
                prefix_len += 1
            else:
                break
        return prefix_len
    
    def pack(self) -> bytes:
        if not self.entries:
            raise ValueError("No files to pack")
        
        self.entries.sort(key=lambda e: e.archive_name.upper())
        
        # Build entry table with placeholders
        entry_table = bytearray()
        prev_name = ""
        
        for idx, entry in enumerate(self.entries):
            name = entry.archive_name
            
            if idx == 0:
                name_bytes = name.encode('latin-1')
                entry_table.extend(struct.pack('<H', len(name_bytes)))
                entry_table.extend(name_bytes)
                entry_table.extend(struct.pack('<I', 0))
                entry_table.extend(struct.pack('<I', entry.size))
            else:
                prefix_len = self._compute_prefix_length(prev_name, name)
                stored = name[prefix_len:]
                stored_bytes = stored.encode('latin-1')
                
                b0 = len(name)
                b1 = prefix_len
                
                entry_table.append(b0)
                entry_table.append(b1)
                entry_table.extend(stored_bytes)
                entry_table.extend(struct.pack('<I', 0))
                entry_table.extend(struct.pack('<I', entry.size))
            
            prev_name = name
        
        entry_table_size = len(entry_table)
        hash_table_size = len(self.entries) * 4
        data_start = self.HEADER_SIZE + entry_table_size + hash_table_size
        
        # Rebuild with correct offsets
        entry_table_patched = bytearray()
        prev_name = ""
        current_offset = data_start
        
        for idx, entry in enumerate(self.entries):
            name = entry.archive_name
            
            if idx == 0:
                name_bytes = name.encode('latin-1')
                entry_table_patched.extend(struct.pack('<H', len(name_bytes)))
                entry_table_patched.extend(name_bytes)
                entry_table_patched.extend(struct.pack('<I', current_offset))
                entry_table_patched.extend(struct.pack('<I', entry.size))
            else:
                prefix_len = self._compute_prefix_length(prev_name, name)
                stored = name[prefix_len:]
                stored_bytes = stored.encode('latin-1')
                
                b0 = len(name)
                b1 = prefix_len
                
                entry_table_patched.append(b0)
                entry_table_patched.append(b1)
                entry_table_patched.extend(stored_bytes)
                entry_table_patched.extend(struct.pack('<I', current_offset))
                entry_table_patched.extend(struct.pack('<I', entry.size))
            
            current_offset += entry.size
            prev_name = name
        
        # Hash table (all zeros - game doesn't use it)
        hash_table = b'\x00' * hash_table_size
        
        # Build header
        header = bytearray()
        header.extend(self.MAGIC)
        header.extend(struct.pack('<I', 0x1A))
        header.extend(b'\x00' * 12)
        header.extend(struct.pack('<I', len(self.entries)))
        header.extend(struct.pack('<I', len(entry_table_patched)))
        
        assert len(header) == self.HEADER_SIZE
        
        # Combine all parts
        output = bytearray()
        output.extend(header)
        output.extend(entry_table_patched)
        output.extend(hash_table)
        
        for entry in self.entries:
            file_data = entry.path.read_bytes()
            output.extend(file_data)
        
        return bytes(output)


# =============================================================================
# High-Level API
# =============================================================================

class WDTArchive:
    """High-level interface for working with WDT archives."""
    
    def __init__(self, wdt_path: Optional[Path] = None, debug: bool = False):
        self.wdt_path = Path(wdt_path) if wdt_path else None
        self.debug = debug
        self.decompressor = LZSSDecompressor()
        self.unpacker: Optional[HMMSYSUnpacker] = None
        self._decompressed_data: Optional[bytes] = None
        self._is_compressed: Optional[bool] = None
    
    def load(self):
        """Load and parse the WDT file."""
        if not self.wdt_path:
            raise ValueError("No WDT file specified")
        
        with open(self.wdt_path, 'rb') as f:
            magic = f.read(4)
            f.seek(0)
            data = f.read()
        
        if magic == b'LZSS':
            self._is_compressed = True
            print(f"Decompressing {self.wdt_path.name}...")
            self._decompressed_data = self.decompressor.decompress(data)
            print(f"Decompressed size: {len(self._decompressed_data):,} bytes")
        elif magic == b'HMMS':
            self._is_compressed = False
            self._decompressed_data = data
            print(f"Loading raw HMMSYS: {self.wdt_path.name} ({len(data):,} bytes)")
        else:
            raise ValueError(f"Unknown format: {magic!r}")
        
        print("Parsing HMMSYS PackFile...")
        self.unpacker = HMMSYSUnpacker(self._decompressed_data, debug=self.debug)
        print(f"Found {len(self.unpacker.entries)} files")
    
    def list_files(self) -> list[HMMSYSFileEntry]:
        if not self.unpacker:
            self.load()
        return list(self.unpacker.list_files())
    
    def extract_all(self, output_dir: Path, verbose: bool = True):
        if not self.unpacker:
            self.load()
        self.unpacker.extract_all(output_dir, verbose=verbose)
    
    def extract_file(self, name: str) -> Optional[bytes]:
        if not self.unpacker:
            self.load()
        for entry in self.unpacker.entries:
            if entry.name.upper() == name.upper():
                return self.unpacker.extract_file(entry)
        return None
    
    @staticmethod
    def create(source_dir: Path, output_path: Path, prefix: str = "",
               validate_bmps: bool = True, fix_bmps: bool = False,
               verbose: bool = True) -> bool:
        """
        Create a WDT archive from a directory.
        
        Args:
            source_dir: Directory containing files to pack
            output_path: Output WDT file path
            prefix: Optional prefix for archive paths
            validate_bmps: Check BMP files for compatibility
            fix_bmps: Auto-convert incompatible BMPs
            verbose: Print progress
        
        Returns:
            True if successful, False if validation failed
        """
        source_dir = Path(source_dir)
        output_path = Path(output_path)
        
        # Validate BMPs if requested
        if validate_bmps:
            if verbose:
                print("Validating BMP files...")
            total, ok, problems = validate_bmps_in_directory(
                source_dir, fix=fix_bmps, verbose=verbose
            )
            
            if problems > 0 and not fix_bmps:
                print(f"\n❌ Found {problems} incompatible BMP file(s).")
                print("   Use --fix-bmp to auto-convert, or convert manually with:")
                print(f"   tzar_wdt.py convert-bmp -d {source_dir}")
                return False
            
            if verbose:
                print(f"\nBMP validation: {ok} OK" + (f", {problems} converted" if fix_bmps and problems else ""))
                print()
        
        # Create archive
        packer = HMMSYSPacker()
        packer.add_directory(source_dir, prefix)
        
        if verbose:
            print(f"Packing {len(packer.entries)} files...")
        
        hmmsys_data = packer.pack()
        
        if verbose:
            print(f"Archive size: {len(hmmsys_data):,} bytes")
        
        output_path.write_bytes(hmmsys_data)
        
        if verbose:
            print(f"Saved to {output_path}")
        
        return True


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_extract(args):
    """Extract files from a WDT archive."""
    archive = WDTArchive(args.archive, debug=args.debug)
    archive.load()
    
    output_dir = Path(args.output)
    print(f"\nExtracting to {output_dir}/...")
    archive.extract_all(output_dir, verbose=not args.quiet)
    print("Done!")
    return 0


def cmd_bundle(args):
    """Create a WDT archive from a directory."""
    source_dir = Path(args.directory)
    output_path = Path(args.output)
    
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        return 1
    
    success = WDTArchive.create(
        source_dir=source_dir,
        output_path=output_path,
        prefix=args.prefix or "",
        validate_bmps=not args.skip_validation,
        fix_bmps=args.fix_bmp,
        verbose=not args.quiet
    )
    
    return 0 if success else 1


def cmd_list(args):
    """List files in a WDT archive."""
    archive = WDTArchive(args.archive, debug=args.debug)
    archive.load()
    
    print("\nFiles in archive:")
    print("-" * 70)
    total_size = 0
    for entry in archive.list_files():
        print(f"{entry.size:>10} bytes  {entry.name}")
        total_size += entry.size
    print("-" * 70)
    print(f"Total: {len(archive.list_files())} files, {total_size:,} bytes")
    return 0


def cmd_convert_bmp(args):
    """Convert BMP files to Tzar-compatible format."""
    if args.directory:
        directory = Path(args.directory)
        if not directory.is_dir():
            print(f"Error: {directory} is not a directory")
            return 1
        
        action = "Checking" if args.check else "Converting"
        print(f"{action} BMP files in {directory}...\n")
        
        total, ok, changed = validate_bmps_in_directory(
            directory, fix=not args.check, verbose=not args.quiet
        )
        
        print(f"\n{'='*50}")
        print(f"Total files:  {total}")
        print(f"Already OK:   {ok}")
        print(f"{'Incompatible' if args.check else 'Converted'}:  {changed}")
        
        return 0 if args.check and changed == 0 else 0
    
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return 1
        
        info = get_bmp_info(input_path)
        if not info['valid']:
            print(f"Error: Invalid BMP file")
            return 1
        
        if args.check:
            if info['is_tzar_compatible']:
                print(f"✓ {input_path}: OK ({info['width']}x{info['height']} 16-bit RGB555)")
                return 0
            else:
                issues = []
                if info['bpp'] != 16:
                    issues.append(f"bpp={info['bpp']}")
                if info['dib_size'] != 40:
                    issues.append(f"header={info['dib_size']}B")
                print(f"✗ {input_path}: INCOMPATIBLE ({', '.join(issues)})")
                return 1
        
        if info['is_tzar_compatible']:
            print(f"✓ {input_path}: Already in correct format")
            return 0
        
        output_path = Path(args.output) if args.output else None
        try:
            convert_bmp_to_rgb555(input_path, output_path)
            out = output_path or input_path
            print(f"✔ Converted: {out}")
            print(f"  {info['width']}x{info['height']}, {info['bpp']}-bit → 16-bit RGB555")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    else:
        print("Error: Specify either an input file or -d/--directory")
        return 1


def cmd_validate(args):
    """Validate assets in a directory."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return 1
    
    print(f"Validating assets in {directory}...\n")
    
    # Validate BMPs
    print("=== BMP Files ===")
    total, ok, problems = validate_bmps_in_directory(directory, fix=False, verbose=not args.quiet)
    
    print(f"\n{'='*50}")
    print(f"BMP files:    {total} total, {ok} OK, {problems} incompatible")
    
    if problems > 0:
        print(f"\n⚠ Found {problems} incompatible BMP file(s).")
        print("  Run with --fix to auto-convert, or use:")
        print(f"  tzar_wdt.py convert-bmp -d {directory}")
        return 1
    
    print("\n✓ All assets validated successfully!")
    return 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='tzar_wdt.py',
        description='Tzar: The Burden of the Crown - WDT Archive Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract an archive
  %(prog)s extract images.wdt -o ./extracted
  
  # List files in an archive
  %(prog)s list images.wdt
  
  # Bundle a directory into a WDT file
  %(prog)s bundle images.wdt -d ./extracted
  
  # Bundle with auto-fix for incompatible BMPs
  %(prog)s bundle images.wdt -d ./extracted --fix-bmp
  
  # Convert a single BMP to Tzar format
  %(prog)s convert-bmp image.bmp
  
  # Check all BMPs in a directory
  %(prog)s convert-bmp -d ./extracted --check
  
  # Validate all assets before bundling
  %(prog)s validate -d ./extracted
        '''
    )
    
    parser.add_argument('--debug', action='store_true', help='Show debug output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract files from a WDT archive')
    extract_parser.add_argument('archive', help='Path to WDT archive')
    extract_parser.add_argument('-o', '--output', default='extracted',
                                help='Output directory (default: extracted)')
    extract_parser.set_defaults(func=cmd_extract)
    
    # Bundle command
    bundle_parser = subparsers.add_parser('bundle', help='Create a WDT archive from a directory')
    bundle_parser.add_argument('output', help='Output WDT file path')
    bundle_parser.add_argument('-d', '--directory', required=True,
                               help='Source directory containing files to pack')
    bundle_parser.add_argument('--prefix', help='Prefix for archive paths (e.g., "IMAGES")')
    bundle_parser.add_argument('--skip-validation', action='store_true',
                               help='Skip BMP format validation')
    bundle_parser.add_argument('--fix-bmp', action='store_true',
                               help='Auto-convert incompatible BMPs to RGB555')
    bundle_parser.set_defaults(func=cmd_bundle)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List files in a WDT archive')
    list_parser.add_argument('archive', help='Path to WDT archive')
    list_parser.set_defaults(func=cmd_list)
    
    # Convert-bmp command
    convert_parser = subparsers.add_parser('convert-bmp',
                                           help='Convert BMP files to Tzar-compatible format')
    convert_parser.add_argument('input', nargs='?', help='Input BMP file')
    convert_parser.add_argument('-o', '--output', help='Output file path')
    convert_parser.add_argument('-d', '--directory',
                                help='Process all BMPs in directory recursively')
    convert_parser.add_argument('--check', action='store_true',
                                help='Check format without converting')
    convert_parser.set_defaults(func=cmd_convert_bmp)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate',
                                            help='Validate assets in a directory')
    validate_parser.add_argument('-d', '--directory', required=True,
                                 help='Directory to validate')
    validate_parser.set_defaults(func=cmd_validate)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
