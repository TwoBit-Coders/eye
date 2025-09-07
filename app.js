 class NavProApp {
            constructor() {
                this.socket = null;
                this.stream = null;
                this.isNavigationActive = false;
                this.frameInterval = null;
                // Get the browser's speech synthesis engine
                this.speechSynth = window.speechSynthesis;
                
                this.initializeElements();
                this.initializeSocket();
                this.initializeCamera();
                this.setupEventListeners();
            }

            // This new function handles all audio output in the browser
            speak(text, priority = 'normal') {
                if (!text || !this.speechSynth) {
                    console.warn("Speech synthesis not available or no text provided.");
                    return;
                }

                // For urgent messages, stop any current speech immediately.
                if (priority === 'urgent' && this.speechSynth.speaking) {
                    this.speechSynth.cancel();
                }
                
                // To prevent a long queue of non-urgent messages, don't speak if already speaking.
                if (this.speechSynth.speaking && priority !== 'urgent') {
                    return; 
                }

                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 4.5; // A comfortable, slightly faster speech rate
                this.speechSynth.speak(utterance);
            }

            initializeElements() {
                this.elements = {
                    startBtn: document.getElementById('startBtn'),
                    stopBtn: document.getElementById('stopBtn'),
                    videoElement: document.getElementById('videoElement'),
                    cameraOverlay: document.getElementById('cameraOverlay'),
                    connectionStatus: document.getElementById('connectionStatus'),
                    statusDot: document.getElementById('statusDot'),
                    statusText: document.getElementById('statusText'),
                    readyMessage: document.getElementById('readyMessage'),
                    captionsContent: document.getElementById('captionsContent'),
                    motionText: document.getElementById('motionText'),
                    motionIndicator: document.getElementById('motionIndicator')
                };
            }

            initializeSocket() {
                this.socket = io();

                this.socket.on('connect', () => this.updateConnectionStatus('connected'));
                this.socket.on('disconnect', () => {
                    this.updateConnectionStatus('disconnected');
                    if (this.isNavigationActive) this.stopNavigation();
                });

                // This handler now also speaks the status message
                this.socket.on('status', (data) => {
                    console.log('Status update:', data.message);
                    this.elements.readyMessage.textContent = data.message;
                    this.speak(data.message);
                });

                // This handler now also speaks the navigation guidance
                this.socket.on('navigation_update', (data) => {
                    this.handleNavigationUpdate(data);
                });

                this.socket.on('frame_result', (data) => this.handleFrameResult(data));
            }

            async initializeCamera() {
                try {
                    this.stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: { ideal: 'environment' } }
                    });
                    this.elements.videoElement.srcObject = this.stream;
                    this.elements.cameraOverlay.style.display = 'none';
                } catch (error) {
                    console.error('Error accessing camera:', error);
                    this.elements.cameraOverlay.innerHTML = `<div>Camera Access Denied</div>`;
                }
            }

            setupEventListeners() {
                this.elements.startBtn.addEventListener('click', () => this.startNavigation());
                this.elements.stopBtn.addEventListener('click', () => this.stopNavigation());
            }

            startNavigation() {
                if (!this.stream) { alert('Camera not available.'); return; }
                this.isNavigationActive = true;
                this.elements.startBtn.disabled = true;
                this.elements.stopBtn.disabled = false;
                this.elements.statusDot.classList.add('running');
                this.elements.statusText.textContent = 'Running';
                this.socket.emit('start_navigation');
                this.frameInterval = setInterval(() => { this.sendFrame(); }, 500); // 2 FPS
            }

            stopNavigation() {
                this.isNavigationActive = false;
                this.elements.startBtn.disabled = false;
                this.elements.stopBtn.disabled = true;
                this.elements.statusDot.classList.remove('running');
                this.elements.statusText.textContent = 'Ready';
                if (this.frameInterval) {
                    clearInterval(this.frameInterval);
                    this.frameInterval = null;
                }
                this.socket.emit('stop_navigation');
            }

            sendFrame() {
                if (!this.isNavigationActive || !this.stream) return;
                const canvas = document.createElement('canvas');
                const video = this.elements.videoElement;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                this.socket.emit('process_frame', { frame: frameData });
            }

            // This function is the core of the synchronized output
            handleNavigationUpdate(data) {
                // 1. Display the caption on screen
                this.elements.captionsContent.textContent = data.message;
                this.elements.captionsContent.className = `captions-content ${data.priority}`;

                // 2. Speak the exact same caption aloud
                this.speak(data.message, data.priority);

                this.updateMotionStatus(data.is_moving, data.stationary_time);
            }

            handleFrameResult(data) {
                if (data.status === 'processed' || data.status === 'skipped') {
                    this.updateMotionStatus(data.is_moving, data.stationary_time);
                } else if (data.status === 'error') {
                    console.error('Frame processing error:', data.message);
                }
            }

            updateMotionStatus(isMoving, stationaryTime) {
                if (isMoving) {
                    this.elements.motionIndicator.classList.remove('still');
                    this.elements.motionText.textContent = 'Moving';
                } else {
                    this.elements.motionIndicator.classList.add('still');
                    this.elements.motionText.textContent = `Still (${stationaryTime.toFixed(1)}s)`;
                }
            }
            
            updateConnectionStatus(status) {
                const isConnected = status === 'connected';
                this.elements.connectionStatus.className = `connection-status ${status}`;
                this.elements.connectionStatus.textContent = isConnected ? 'Connected' : 'Disconnected';
                this.elements.statusDot.classList.toggle('connected', isConnected);
                if (!isConnected) this.elements.statusDot.classList.remove('running');
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => { new NavProApp(); });
         