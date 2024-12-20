function findAreas(grid, skipNumbers = []) {
  const rows = grid.length;
  const cols = grid[0].length;
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false));
  const areas = [];

  // Convert skipNumbers array to a Set for efficient lookups
  const skipSet = new Set(skipNumbers);

  // Directions for neighboring cells: up, down, left, right
  const directions = [
    [-1, 0], [1, 0], [0, -1], [0, 1]
  ];

  // Perform a DFS to find all connected cells with the same number
  function dfs(r, c, number, area) {
    area.squares.push([r, c]);
    visited[r][c] = true;

    for (const [dr, dc] of directions) {
      const nr = r + dr;
      const nc = c + dc;
      if (
          nr >= 0 && nr < rows &&
          nc >= 0 && nc < cols &&
          !visited[nr][nc] &&
          grid[nr][nc] === number &&
          !skipSet.has(grid[nr][nc])
      ) {
        dfs(nr, nc, number, area);
      }
    }
  }

  // Iterate over the grid to find all unvisited areas
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!visited[r][c] && !skipSet.has(grid[r][c])) {
        const area = { groupId: grid[r][c], squares: [] };
        dfs(r, c, grid[r][c], area);
        areas.push(area);
      } else {
        visited[r][c] = true; // Mark skipped numbers as visited
      }
    }
  }

  return areas.filter(area => area.squares.length > 1);
}

function findBoundingBoxes(areas) {
  return areas.map(area => {
    let minRow = Infinity, maxRow = -Infinity;
    let minCol = Infinity, maxCol = -Infinity;

    // Iterate over all the points in the area to find the bounds
    for (const [row, col] of area.squares) {
      minRow = Math.min(minRow, row);
      maxRow = Math.max(maxRow, row);
      minCol = Math.min(minCol, col);
      maxCol = Math.max(maxCol, col);
    }

    // Return the bounding box as [top-left, bottom-right]
    return {
      groupId: area.groupId,
      topLeft: [minRow, minCol],
      bottomRight: [maxRow, maxCol]
    };
  });
}

function drawBoundingBoxes(boundingBoxes, canvasId) {
  const color = '#ff0000';
  boundingBoxes.forEach(box => drawBoundingBox(box, canvasId, color));
}

function drawBoundingBox(box, canvasId, color = '#ff0000') {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const minSide = Math.min(canvas.width, canvas.height);

  const cellSize = minSide / 7; // Size of each grid cell in pixels
  const borderThickness = Math.round(Math.max(5, Math.min(11, cellSize / 16)));

  // Set the stroke style
  ctx.strokeStyle = color;
  ctx.lineWidth = borderThickness;

  const [topRow, leftCol] = box.topLeft;
  const [bottomRow, rightCol] = box.bottomRight;

  // give each box a slight offset such that the boxes' edges dont overlap 100%
  const offset = 0; // Math.floor(20 * Math.random()) - 10;

  // Calculate rectangle dimensions, also make sure it stays in canvas bounds
  let x = Math.round(leftCol * cellSize) + offset;
  x = Math.max(x, borderThickness / 2);
  let y = Math.round(topRow * cellSize) + offset;
  y = Math.max(y, borderThickness / 2);
  let width = Math.round((rightCol - leftCol + 1) * cellSize) + offset;
  width = Math.min(width, minSide - x - borderThickness / 2);
  let height = Math.round((bottomRow - topRow + 1) * cellSize) + offset;
  height = Math.min(height, minSide - y - borderThickness / 2);

  // Draw the rectangle
  ctx.strokeRect(x, y, width, height);

  // Draw group name
  const groupName = grouper.getGroupName(box.groupId);
  const fontSize = Math.max(16, minSide / 25);
  const textX = x + borderThickness / 2 + 2;
  const textY = y + borderThickness / 2 + 2;
  const maxWidth = width - borderThickness - 4;
  ctx.textBaseline = "top";
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 3;
  ctx.fillStyle = color;
  ctx.strokeText(groupName, textX, textY, maxWidth);
  ctx.fillText(groupName, textX, textY, maxWidth);
}

function to2DArray(array, rows, cols) {
  if (array.length !== rows * cols) {
    throw new Error("The given array size does not match the specified dimensions.");
  }

  const result = [];
  for (let i = 0; i < rows; i++) {
    // Slice a chunk of `cols` from the array for each row
    result.push(array.slice(i * cols, i * cols + cols));
  }
  return result;
}
