var TWO_PI = Math.PI*2;
var WIN_AMOUNT = 7;
var RENDER_GLOBAL = false;
var RENDER_ENV = false;
var OPPONENT_INDEX = undefined;

function start(random) {
  slimeLeftScore = 0;
  slimeRightScore = 0;

  slimeLeft.img = greenSlimeImage;
  if(OPPONENT_INDEX !== undefined) {
    console.log("Boo: " + slimeAIs)
    var slimeAIProps = slimeAIs[OPPONENT_INDEX];
    //slimeRight.color = slimeAIProps.color;
    //legacySkyColor   = slimeAIProps.legacySkyColor;
    //backImage        = backImages[slimeAIProps.backImageName];
    //backTextColor    = slimeAIProps.backTextColor;
    //legacyGroundColor= slimeAIProps.legacyGroundColor;
    //legacyBallColor  = slimeAIProps.legacyBallColor;
    //newGroundColor   = slimeAIProps.newGroundColor;

    slimeAI          = newSlimeAI(false,slimeAIProps.name);
    slimeAIProps.initAI(slimeAI);
  } else {
    slimeAI          = null;
  }

  legacySkyColor   = '#00f';
  backImage        = backImages['sky'];
  backTextColor    = '#000';
  legacyGroundColor= '#888';
  legacyBallColor  = '#fff';
  newGroundColor   = '#ca6';

  starting = random >= 0.5;
  initRound(starting);

  updatesToPaint = 0;
  updateCount = 0;
  gameState = GAME_STATE_RUNNING
  renderBackground(); // clear the field
  canvas.style.display = 'block';
  menuDiv.style.display = 'none';
  //gameIntervalObject = setInterval(asyncStep, 20);
}

function getPixels() {
  return canvas.toDataURL('image/png').substring(21);
}

function asyncStep() {
  action1 = [keysDown[KEY_A], keysDown[KEY_W], keysDown[KEY_D]];
  action2 = [keysDown[KEY_LEFT], keysDown[KEY_UP], keysDown[KEY_RIGHT]];
  keysDown = {};
  step(action1, action2, false, 1);
}

function renderBackground() {
  ctx.fillStyle=legacySkyColor;
  ctx.fillRect(0, 0, viewWidth, courtYPix);
  ctx.fillStyle = legacyGroundColor;
  ctx.fillRect(0, courtYPix, viewWidth, viewHeight - courtYPix);
  ctx.fillStyle='#fff'
  ctx.fillRect(viewWidth/2-2,7*viewHeight/10,4,viewHeight/10+5);
  // render scores
  renderPoints(slimeLeftScore, 30, 40);
  renderPoints(slimeRightScore, viewWidth - 30, -40);
}

function renderPoints(score, initialX, xDiff) {
  ctx.fillStyle = '#ff0';
  var x = initialX;
  for(var i = 0; i < score; i++) {
    ctx.beginPath();
    ctx.arc(x, 25, 12, 0, TWO_PI);
    ctx.fill();
    x += xDiff;
  }
  ctx.strokeStyle = backTextColor;
  ctx.lineWidth = 2;
  x = initialX;
  for(var i = 0; i < WIN_AMOUNT; i++) {
    ctx.beginPath();
    ctx.arc(x, 25, 12, 0, TWO_PI);
    ctx.stroke();
    x += xDiff;
  }
}

function initRound(server) {
  ball.x = server ? 200 : 800;
  ball.y = 356;
  ball.velocityX = 0;
  ball.velocityY = 0;

  slimeLeft.x = 200;
  slimeLeft.y = 0;
  slimeLeft.velocityX = 0;
  slimeLeft.velocityY = 0;

  slimeRight.x = 800;
  slimeRight.y = 0;
  slimeRight.velocityX = 0;
  slimeRight.velocityY = 0;
}

function renderGame() {
  if(updatesToPaint == 0) {
    console.log("ERROR: render called but not ready to paint");
  } else {
    if(updatesToPaint > 1) {
      console.log("WARNING: render missed " + (updatesToPaint - 1) + " frame(s)");
    }
    renderBackground();
    ctx.fillStyle = '#000';
    //ctx.font = "20px Georgia";
    //ctx.fillText("Score: " + slimeLeftScore, 140, 20);
    //ctx.fillText("Score: " + slimeRightScore, viewWidth - 230, 20);
    ball.render();
    slimeLeft.render();
    slimeRight.render();
    updatesToPaint = 0;
  }
}

function renderEndOfPoint() {
  var textWidth = ctx.measureText(endOfPointText).width;
  renderGame();
  ctx.fillStyle = '#000';
  ctx.fillText(endOfPointText,
    (viewWidth - textWidth)/2, courtYPix + (viewHeight - courtYPix)/2);
}

function reset(random) {
  start(random);
  return step([0,0,0],[0,0,0], 1);
}

// player actons are boolean array for left, up, right movement
function step(player1Action, player2Action, numFrames) {
  console.log("action: " + player2Action);
  var reward = 0;
  var done = false;
  for (var i = 0; i < numFrames; i++) {
    if (player1Action !== null) {
      leftKey1 = player1Action[0];
      upKey1 = player1Action[1];
      rightKey1 = player1Action[2];
      keysDown[KEY_A] = leftKey1;
      keysDown[KEY_W] = upKey1;
      keysDown[KEY_D] = rightKey1;
    }

    if (player2Action !== null) {
      leftKey2 = player2Action[0];
      upKey2 = player2Action[1];
      rightKey2 = player2Action[2];
      keysDown[KEY_LEFT] = leftKey2;
      keysDown[KEY_UP] = upKey2;
      keysDown[KEY_RIGHT] = rightKey2;
    }
    // Only render last frame
    reward += gameIteration((RENDER_ENV && i === numFrames - 1));
    done = slimeLeftScore >= WIN_AMOUNT || slimeRightScore >= WIN_AMOUNT;
    if (done) {
      break;
    }
  }

  return {
    "player1": [
      (slimeLeft.x - 500) / 500,
      slimeLeft.y / 705,
      slimeLeft.velocityX / 8,
      slimeLeft.velocityY / 31,
      slimeLeftScore / 7
    ],
    "player2": [
      (slimeRight.x - 500) / 500,
      slimeRight.y / 705,
      slimeRight.velocityX / 8,
      slimeRight.velocityY / 31,
      slimeRightScore / 7
    ],
    "ball": [
      (ball.x - 500) / 500,
      ball.y / 800, //hacky max height no way it goes over 100 over the slime
      ball.velocityX / MAX_VELOCITY_X,
      ball.velocityY / MAX_VELOCITY_Y
    ],
    "reward": reward,
    "done": done
  }
}
// returns true if end of point
function gameIteration(render) {
  if(gameState == GAME_STATE_RUNNING) {
    updateCount++;
    // if(slowMotion && (updateCount % 2) == 0)
      // return;
    if(updatesToPaint > 0) {
      console.log("WARNING: updating frame before it was rendered");
    }

    var reward = updateFrame();
    updatesToPaint++;
    if((RENDER_GLOBAL || render) && updatesToPaint > 1) {
      //requestAnimationFrame(renderGame);
      renderGame();
    }
  }
  return reward;
}

// returns true if end of point
function updateFrame() {
  if(OPPONENT_INDEX !== undefined) {
    slimeAI.move(false); // Move the right slime
    updateSlimeVelocitiesWithDoubleKeys(slimeLeft,
      KEY_A,KEY_LEFT,
      KEY_D,KEY_RIGHT,
      KEY_W,KEY_UP);
    updateSlimeVelocities(slimeRight,slimeAI.movement,slimeAI.jumpSet);
  } else {
    updateSlimeVelocitiesWithKeys(slimeLeft, KEY_A,KEY_D,KEY_W);
    updateSlimeVelocitiesWithKeys(slimeRight, KEY_LEFT,KEY_RIGHT,KEY_UP);
  }

  updateSlime(slimeLeft, 0, 1000);
  updateSlime(slimeRight, 0, 1000);

  // Allows slimes to go accross the net
  //updateSlime(slimeLeft, 0, 1000);
  //updateSlime(slimeRight, 0, 1000);

  return updateBall();
}

// Game Update Functions
function updateSlimeVelocities(s,movement,jump) {
  if(movement == 0) {
    s.velocityX = 0;
  } else if(movement == 1) {
    s.velocityX = -8;
  } else if(movement == 2) {
    s.velocityX = 8;
  } else {
    throw "slime movement " + movement + " is invalid";
  }
  if(jump && s.velocityY <= 3) {
    s.velocityY = 31;
  }
}

function updateSlimeVelocitiesWithKeys(s,left,right,up) {
  // update velocities
  if(keysDown[left]) {
    if(keysDown[right]) {
      s.velocityX = 0;
    } else {
      s.velocityX = -8;
    }
  } else if(keysDown[right]) {
    s.velocityX = 8;
  } else {
    s.velocityX = 0;
  }
  if(s.velocityY <= 3 && keysDown[up]) {
    keysDown[up] = false;
    s.velocityY = 31;
  }
}
function updateSlimeVelocitiesWithDoubleKeys(s,left1,left2,right1,right2,up1,up2) {
  // update velocities
  if(keysDown[left1] || keysDown[left2]) {
    if(keysDown[right1] || keysDown[right2]) {
      s.velocityX = 0;
    } else {
      s.velocityX = -8;
    }
  } else if(keysDown[right1] || keysDown[right2]) {
    s.velocityX = 8;
  } else {
    s.velocityX = 0;
  }
  if(s.velocityY <= 3 && (keysDown[up1] || keysDown[up2])) {
    keysDown[up1] = false;
    keysDown[up2] = false;
    s.velocityY = 31;
  }
}
function updateSlime(s, leftLimit, rightLimit) {
  if(s.velocityX != 0) {
    s.x += s.velocityX;
    if(s.x < leftLimit) s.x = leftLimit;
    else if(s.x > rightLimit) s.x = rightLimit;
  }
  if(s.velocityY != 0 || s.y > 0) {
    s.velocityY -= 2;
    s.y += s.velocityY;
    if(s.y < 0) {
      s.y = 0;
      s.velocityY = 0;
    }
  }

  if (s.x >= 460 && s.x <= 540 && s.y <= 180) {
    if (s.velocityY <= 0 && s.y >= 125) {
      s.velocityY = 0;
      s.y = 130;
    } else if (s.x < 500) { // hits side of net
      s.x = 460;
      s.velocityX = 0;
    } else {
      s.x = 540;
      s.velocityX = 0;
    }
  }

  // add ceiling
  if (s.y > 705) {
    s.y = 705;
  }
}

var MAX_VELOCITY_X = 15;
var MAX_VELOCITY_Y = 22;
function collisionBallSlime(s) {
  var dx = 2 * (ball.x - s.x);
  var dy = ball.y - s.y;
  var dist = Math.trunc(Math.sqrt(dx * dx + dy * dy));

  var dVelocityX = ball.velocityX - s.velocityX;
  var dVelocityY = ball.velocityY - s.velocityY;

  if(dy > 0 && dist < ball.radius + s.radius && dist > FUDGE) {
    var oldBall = {x:ball.x,y:ball.y,velocityX:ball.velocityX,velocityY:ball.velocityY};
    ball.x = s.x + Math.trunc(Math.trunc((s.radius + ball.radius) / 2) * dx / dist);
    ball.y = s.y + Math.trunc((s.radius + ball.radius) * dy / dist);

    var something = Math.trunc((dx * dVelocityX + dy * dVelocityY) / dist);

    if(something <= 0) {
      ball.velocityX += Math.trunc(s.velocityX - 2 * dx * something / dist);
      ball.velocityY += Math.trunc(s.velocityY - 2 * dy * something / dist);
      if(     ball.velocityX < -MAX_VELOCITY_X) ball.velocityX = -MAX_VELOCITY_X;
      else if(ball.velocityX >  MAX_VELOCITY_X) ball.velocityX =  MAX_VELOCITY_X;
      if(     ball.velocityY < -MAX_VELOCITY_Y) ball.velocityY = -MAX_VELOCITY_Y;
      else if(ball.velocityY >  MAX_VELOCITY_Y) ball.velocityY =  MAX_VELOCITY_Y;
    }
  }
}

var FUDGE = 5; // not sure why this is needed
// returns true if end of point
function updateBall() {
  ball.velocityY += -1; // gravity
  if(ball.velocityY < -MAX_VELOCITY_Y) {
    ball.velocityY = -MAX_VELOCITY_Y;
  }

  ball.x += ball.velocityX;
  ball.y += ball.velocityY;

  collisionBallSlime(slimeLeft);
  collisionBallSlime(slimeRight);

  // handle wall hits
  if (ball.x < 15) {
    ball.x = 15;
    ball.velocityX = -ball.velocityX;
  } else if(ball.x > 985){
    ball.x = 985;
    ball.velocityX = -ball.velocityX;
  }
  // hits the post
  if (ball.x > 480 && ball.x < 520 && ball.y < 140) {
    // bounces off top of net
    if (ball.velocityY < 0 && ball.y > 130) {
      ball.velocityY *= -1;
      ball.y = 130;
    } else if (ball.x < 500) { // hits side of net
      ball.x = 480;
      ball.velocityX = ball.velocityX >= 0 ? -ball.velocityX : ball.velocityX;
    } else {
      ball.x = 520;
      ball.velocityX = ball.velocityX <= 0 ? -ball.velocityX : ball.velocityX;
    }
  }

  var reward = 0.0;
  // Check for end of point
  if(ball.y < 0) {
    if(ball.x > 500) {
      leftWon = true;
      reward = 1.0;
      slimeLeftScore++;
    } else {
      leftWon = false;
      reward = -1.0;
      slimeRightScore++;
    }
    endPoint()
  }
  return reward;
}

function endMatch() {
  gameState = GAME_STATE_SHOW_WINNER;
  clearInterval(gameIntervalObject);
  if(OPPONENT_INDEX !== undefined) {
    if(leftWon) {
nextSlimeIndex++;
if(nextSlimeIndex >= slimeAIs.length) {
        menuDiv.innerHTML = '<div style="text-align:center;">' +
          '<h1 style="margin:50px 0 20px 0;">You beat everyone!</h1>' +
          "Press 'space' to do it all over again!" +
          '</div>';
} else {
        menuDiv.innerHTML = '<div style="text-align:center;">' +
          '<h1 style="margin:50px 0 20px 0;">You won!</h1>' +
          "Press 'space' to continue..." +
          '</div>';
}
    } else {
nextSlimeIndex = 0;
      menuDiv.innerHTML = '<div style="text-align:center;">' +
        '<h1 style="margin:50px 0 20px 0;">You lost :(</h1>' +
        "Press 'space' to retry..." +
        '</div>';
    }
  } else {
    menuDiv.innerHTML = '<div style="text-align:center;">' +
      '<h1 style="margin:50px 0 20px 0;">Player ' + (leftWon ? '1' : '2') + ' Wins!</h1>' +
      "Press 'space' for rematch..." +
      '</div>';
  }
  menuDiv.style.display = 'block';
  canvas.style.display = 'none';
}
function startNextPoint() {
  initRound(leftWon);
  updatesToPaint = 0;
  updateCount = 0;
  gameState = GAME_STATE_RUNNING;
}

function endPoint() {
  if(slimeAI) {
    keysDown[KEY_UP] = false;
    keysDown[KEY_RIGHT] = false;
    keysDown[KEY_LEFT] = false;
  }

  if(slimeLeftScore >= WIN_AMOUNT) {
    endMatch(true);
    return;
  }
  if(slimeRightScore >= WIN_AMOUNT) {
    endMatch(false);
    return;
  }

  endOfPointText = 'Player ' + (leftWon ? '1':'2') + ' scores!';
  gameState = GAME_STATE_POINT_PAUSE;
  if (RENDER_ENV) {
    renderEndOfPoint();
  }
  startNextPoint();
  //setTimeout(function () {
  //  if(gameState == GAME_STATE_POINT_PAUSE) {
  //    startNextPoint();
  //  }
  //}, 700);
}

