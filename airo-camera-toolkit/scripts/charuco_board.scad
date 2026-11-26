/*
Two-color Charuco board generator for the AIRO default calibration board.

The model is split into two printable bodies:
- `part = "white"`: a white base plate with the black regions cut out.
- `part = "black"`: the black regions that fill those cutouts.
- `part = "combined"`: preview both bodies together.

The embedded marker layout matches the default AIRO/OpenCV Charuco board used
in `airo_camera_toolkit.calibration.fiducial_markers`:
- 7 x 5 squares
- 40 mm square size
- 31 mm marker size
- OpenCV DICT_4X4_250

Export the white and black bodies separately if your slicer or printer needs
distinct STLs for two-color printing.
*/

part = "combined"; // "white", "black", or "combined"

// Physical board parameters.
squares_x = 7;
squares_y = 5;
square_size_mm = 40;
marker_size_mm = 31;
board_thickness_mm = 2.0;

// Small overlap so the two colors fuse cleanly in the slicer.
fit_overlap_mm = 0.05;

// Optional tool-flange mounting pattern.
// Set to "iso_9409_1_50_4_m6" for a common 50 mm, 4-bolt M6 interface.
mount_pattern = "iso_9409_1_50_4_m6"; // "none" or "iso_9409_1_50_4_m6"
mount_hole_spacing_mm = 50;
mount_hole_diameter_mm = 6.6;
mount_center_hole_diameter_mm = 0;
mount_extension_width_mm = 80;
mount_extension_length_mm = 80;

// If true, center the board around the origin.
center_board = true;

// The default Charuco board places markers in the white squares, in row-major
// order matching OpenCV's `board.getIds()` output.
marker_grid_size = 6;
marker_rows = [
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
  ],
];

function board_width_mm() = squares_x * square_size_mm;
function board_height_mm() = squares_y * square_size_mm;
function board_origin_mm() = center_board ? [-board_width_mm() / 2, -board_height_mm() / 2, 0] : [0, 0, 0];
function marker_margin_mm() = (square_size_mm - marker_size_mm) / 2;
function marker_cell_size_mm() = marker_size_mm / marker_grid_size;

function is_black_square(row, col) = (row + col) % 2 == 0;

// White-square markers are laid out row-major in the same order as
// OpenCV's board.getIds() for the default 7x5 board.
function markers_before_row(row) =
    (row % 2 == 0)
    ? floor(row / 2) * squares_x
    : floor(row / 2) * squares_x + floor(squares_x / 2);

function marker_index_for_square(row, col) = markers_before_row(row) + floor(col / 2);

function marker_cell_is_black(marker_index, marker_row, marker_col) =
  marker_rows[marker_index][marker_row][marker_col] == 1;

function mount_extension_body_length_mm() = mount_extension_length_mm - mount_extension_width_mm / 2;

module mount_holes() {
  if (mount_pattern == "iso_9409_1_50_4_m6") {
    for (x_sign = [-1, 1]) {
      for (y_sign = [-1, 1]) {
        translate([
          x_sign * mount_hole_spacing_mm / 2,
          -board_height_mm() / 2 - mount_extension_length_mm / 2 + y_sign * mount_hole_spacing_mm / 2,
          -fit_overlap_mm,
        ])
          cylinder(h = board_thickness_mm + 2 * fit_overlap_mm, d = mount_hole_diameter_mm, center = false, $fn = 64);
      }
    }

    if (mount_center_hole_diameter_mm > 0) {
      translate([0, 0, -fit_overlap_mm])
        cylinder(
          h = board_thickness_mm + 2 * fit_overlap_mm,
          d = mount_center_hole_diameter_mm,
          center = false,
          $fn = 64
        );
    }
  }
}

module mount_extension() {
  if (mount_pattern == "iso_9409_1_50_4_m6") {
    translate([0, 0, 0])  //board_width_mm() / 2
      union() {
        translate([board_width_mm() / 2, -mount_extension_body_length_mm()/2, 0]) //mount_extension_width_mm / 2
          cube([mount_extension_width_mm, mount_extension_body_length_mm(), board_thickness_mm], center = true);
        translate([board_width_mm() / 2, -mount_extension_body_length_mm(), 0]) // mount_extension_body_length_mm()
          cylinder(h = board_thickness_mm, r = mount_extension_width_mm / 2, center = true, $fn = 96);
      }
  }
}

module black_geometry_core() {
  for (row = [0 : squares_y - 1]) {
    for (col = [0 : squares_x - 1]) {
      if (is_black_square(row, col)) {
        black_square(row, col);
      } else {
        marker_black_cells(row, col, marker_index_for_square(row, col));
      }
    }
  }
}

module board_base() {
  union() {
    cube([board_width_mm(), board_height_mm(), board_thickness_mm], center = false);
    mount_extension();
  }
}

module black_square(row, col) {
  translate([
    col * square_size_mm - fit_overlap_mm / 2,
    row * square_size_mm - fit_overlap_mm / 2,
    -fit_overlap_mm,
  ])
    cube(
      [
        square_size_mm + fit_overlap_mm,
        square_size_mm + fit_overlap_mm,
        board_thickness_mm + 2 * fit_overlap_mm,
      ],
      center = false
    );
}

module marker_black_cells(row, col, marker_index) {
  for (marker_row = [0 : marker_grid_size - 1]) {
    for (marker_col = [0 : marker_grid_size - 1]) {
      if (marker_cell_is_black(marker_index, marker_row, marker_col)) {
        translate([
          col * square_size_mm + marker_margin_mm() + marker_col * marker_cell_size_mm() - fit_overlap_mm / 2,
          row * square_size_mm + marker_margin_mm() + marker_row * marker_cell_size_mm() - fit_overlap_mm / 2,
          -fit_overlap_mm,
        ])
          cube(
            [
              marker_cell_size_mm() + fit_overlap_mm,
              marker_cell_size_mm() + fit_overlap_mm,
              board_thickness_mm + 2 * fit_overlap_mm,
            ],
            center = false
          );
      }
    }
  }
}

module black_geometry() {
  black_geometry_core();
}

module white_body() {
  difference() {
    board_base();
    black_geometry_core();
    mount_holes();
  }
}

module charuco_board(selected_part = "black") {
  translate(board_origin_mm()) {
    if (selected_part == "white") {
      color("white") white_body();
    } else if (selected_part == "black") {
      color("black") black_geometry();
    } else {
      color("white") white_body();
      color("black") black_geometry();
    }
  }
}

charuco_board("black");