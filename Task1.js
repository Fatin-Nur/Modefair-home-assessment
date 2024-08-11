const points = {
    'A': { x: 0, y: 0 }, 'B': { x: 1, y: 0 }, 'C': { x: 2, y: 0 },
    'D': { x: 0, y: 1 }, 'E': { x: 1, y: 1 }, 'F': { x: 2, y: 1 },
    'G': { x: 0, y: 2 }, 'H': { x: 1, y: 2 }, 'I': { x: 2, y: 2 }
  };
  
  function canConnect(a, b, visited) {
    const ax = points[a].x, ay = points[a].y;
    const bx = points[b].x, by = points[b].y;
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    const midPoint = Object.keys(points).find(key => points[key].x === mx && points[key].y === my);
    return (
      ax === bx || ay === by || Math.abs(ax - bx) === Math.abs(ay - by) // Check for straight line
    ) && (!midPoint || visited.includes(midPoint)); // Check if midpoint is already visited
  }
  
  function searchPatterns(current, path, visited, patterns, target, depth) {
    if (path.length === depth) {
      if (current === target) patterns.push(path.join(''));
      return;
    }
    for (let next of Object.keys(points)) {
      if (!visited.includes(next) && canConnect(current, next, visited)) {
        searchPatterns(next, [...path, next], [...visited, next], patterns, target, depth);
      }
    }
  }
  
  function listPatterns(first, second, third) {
    let patterns = [];
    searchPatterns(first, [first], [first], patterns, second, 3); // Find paths from first to second
    let validPatterns = [];
    patterns.forEach(pattern => {
      let path = pattern.split('');
      searchPatterns(second, path, path, validPatterns, third, path.length + 4);
    });
    return validPatterns;
  }
  
  // Example Usage
  console.log(listPatterns("A", "I", "C"));
  