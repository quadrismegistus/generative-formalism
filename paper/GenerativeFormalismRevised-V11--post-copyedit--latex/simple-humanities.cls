%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMPLE HUMANITIES CLASS FILE
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{simple-humanities}[Simple Humanities Paper Format]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASS OPTIONS
\newif\if@anonymous
\@anonymousfalse
\DeclareOption{anon}{\@anonymoustrue}
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASE CLASS - 12pt TIMES NEW ROMAN
\LoadClass[12pt,letterpaper]{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ESSENTIAL PACKAGES
\RequirePackage{times}           % Times New Roman font
\RequirePackage[T1]{fontenc}     % Font encoding
\RequirePackage{setspace}        % Double spacing
\RequirePackage[
  top=1in,
  bottom=1in,
  left=1in,
  right=1in
]{geometry}                      % Standard 1-inch margins
\RequirePackage{graphicx}        % For images
\RequirePackage{longtable}       % For long tables
\RequirePackage{array}           % Table enhancements
\RequirePackage{booktabs}        % Better table rules
\RequirePackage{caption}         % For caption formatting
\RequirePackage{float}           % For precise figure placement
\RequirePackage[hidelinks,bookmarks=true,bookmarksopen=true,bookmarksnumbered=true]{hyperref}  % Hyperlinks without boxes + bookmarks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BIBLIOGRAPHY SETUP
\RequirePackage[
  style=mla,
  backend=biber
]{biblatex}
\RequirePackage[american]{babel}
\RequirePackage{csquotes}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CAPTION FORMATTING
\captionsetup{
  font=small,          % Base font size for all caption elements
  labelfont={bf,it,small},    % Makes "Figure 1:" bold, italic, and small
  textfont={it,small},        % Makes caption text italic and small
  justification=justified,     % Justified text alignment
  singlelinecheck=false,       % Applies justification even to single-line captions
  margin=0pt,          % No extra margin
  skip=10pt,           % Space between caption and figure/table
  labelsep=period      % Changes "Figure 1:" to "Figure 1."
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% METADATA COMMANDS
\newcommand{\@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\@keywords}{#1}}
\newcommand{\@subtitle}{}
\newcommand{\subtitle}[1]{\renewcommand{\@subtitle}{#1}}
\newcommand{\@pubyear}{}
\newcommand{\pubyear}[1]{\renewcommand{\@pubyear}{#1}}
\newcommand{\@pubvolume}{}
\newcommand{\pubvolume}[1]{\renewcommand{\@pubvolume}{#1}}
\newcommand{\@pagestart}{}
\newcommand{\pagestart}[1]{\renewcommand{\@pagestart}{#1}}
\newcommand{\@pageend}{}
\newcommand{\pageend}[1]{\renewcommand{\@pageend}{#1}}
\newcommand{\@conferencename}{}
\newcommand{\conferencename}[1]{\renewcommand{\@conferencename}{#1}}
\newcommand{\@conferenceeditors}{}
\newcommand{\conferenceeditors}[1]{\renewcommand{\@conferenceeditors}{#1}}
\newcommand{\@doi}{}
\newcommand{\doi}[1]{\renewcommand{\@doi}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUTHOR AND AFFILIATION SYSTEM
\RequirePackage{xparse}
\RequirePackage{etoolbox}

\newcounter{authorcount}
\setcounter{authorcount}{0}
\def\@authornames{}
\def\@authornamesfull{}

\define@key{author}{orcid}{\def\@currentorcid{#1}}
\setkeys{author}{orcid={}}

\RenewDocumentCommand{\author}{O{} m O{}}{%
  \stepcounter{authorcount}
  \setkeys{author}{#3}  
  
  \ifnum\value{authorcount}=1
    \def\@authornames{#2}
    \ifx\@currentorcid\@empty
      \def\@authornamesfull{\mbox{#2}}
    \else
      \def\@authornamesfull{\mbox{#2}}
    \fi
  \else
    \appto\@authornames{, \mbox{#2}}
    \ifx\@currentorcid\@empty
      \appto\@authornamesfull{, \mbox{#2}\textsuperscript{#1}}
    \else
      \appto\@authornamesfull{, \mbox{#2}\textsuperscript{#1}}
    \fi
  \fi
}

\newcounter{affilcount}
\setcounter{affilcount}{0}

\newcommand{\affiliation}[2]{%
  \stepcounter{affilcount}
  \ifnum\value{authorcount}=1
    \ifnum\value{affilcount}=1
      \def\@affillist{#2}
    \else
      \appto\@affillist{\\\textsuperscript{#1}~#2}
    \fi
  \else
    \ifnum\value{affilcount}=1
      \def\@affillist{\textsuperscript{#1}~#2}
    \else
      \appto\@affillist{\\\textsuperscript{#1}~#2}
    \fi
  \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ABSTRACT ENVIRONMENT
\renewenvironment{abstract}{%
  \begin{center}
    \bfseries Abstract
  \end{center}
  \vspace{0.5em}
  \small
  \noindent\ignorespaces
}{
  
  \noindent\textbf{Keywords:} \@keywords
  \vspace{1em}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOUBLE SPACING FOR ENTIRE DOCUMENT
\doublespacing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SINGLE SPACING FOR QUOTES
\BeforeBeginEnvironment{quote}{\singlespacing}
\AfterEndEnvironment{quote}{\doublespacing}

\BeforeBeginEnvironment{quotation}{\singlespacing}
\AfterEndEnvironment{quotation}{\doublespacing}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TITLE PAGE FORMATTING
\renewcommand{\maketitle}{%
  \begin{center}
    {\large\bfseries\@title}
    \ifx\@subtitle\@empty\else
      \\
      {\large\bfseries\@subtitle}
    \fi
    \vspace{1em}
    
    \if@anonymous
      {\normalsize Anonymous Submission}
      \vspace{1em}
    \else
      {\normalsize \@authornamesfull}
      % \vspace{0.25em}
      
      {\small \@affillist}
      % \vspace{0.5em}
    \fi
    
    % {\normalsize \@date}
  \end{center}
  \vspace{2em}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTION FORMATTING (SIMPLE)
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
  {-2ex \@plus -1ex \@minus -.2ex}%
  {1ex \@plus .2ex}%
  {\normalfont\large\bfseries}}

\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
  {-1.5ex \@plus -1ex \@minus -.2ex}%
  {0.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAGRAPH INDENTATION
\setlength{\parindent}{0.5in}    % Standard paragraph indent
\setlength{\parskip}{0pt}        % No extra space between paragraphs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PAGE NUMBERING (TOP RIGHT)
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}

% No page number on first page
\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BIBLIOGRAPHY FORMATTING
\renewenvironment{thebibliography}[1]{%
  \section*{\refname}%
  \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
  \list{}%
       {\setlength{\leftmargin}{0.5in}%
        \setlength{\itemindent}{-0.5in}%
        \setlength{\listparindent}{\itemindent}%
        \setlength{\itemsep}{0.5em}%
        \setlength{\parsep}{0pt}%
        \setlength{\labelwidth}{0pt}%
        \setlength{\labelsep}{0pt}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
 {\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
\renewcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\renewcommand{\bibitem}[1]{\item \hangindent=0.5in}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOOTNOTE FORMATTING
\renewcommand{\footnoterule}{\vspace*{-3pt}\hrule width 2in height 0.4pt \vspace*{2.6pt}}

% Remove footnote indentation and add space between footnotes
\renewcommand{\@makefntext}[1]{%
\raggedright
  \noindent
  \textsuperscript{\@thefnmark}\hspace{0.5em}%
  #1%
  \vspace{0.5em}% Add vertical space after each footnote
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CUSTOM FIGURE COMMAND
% Usage: \easyfig{filename}{caption}{label}
% Example: \easyfig{media/image1.png}{Your caption here}{fig:mylabel}
\newcommand{\easyfig}[3]{%
  \begin{figure}[H]
    \centering
    \includegraphics[width=1.0\linewidth]{#1}
    \caption{#2}
    \label{#3}
  \end{figure}%
  \noindent%
}

% Alternative with width control:
% Usage: \easyfigwidth{filename}{width}{caption}{label}
% Example: \easyfigwidth{media/image1.png}{0.8\linewidth}{Your caption here}{fig:mylabel}
\newcommand{\easyfigwidth}[4]{%
  \begin{figure}[H]
    \centering
    \includegraphics[width=#2]{#1}
    \caption{#3}
    \label{#4}
  \end{figure}%
  \noindent%
}

% Advanced version with optional width parameter:
% Usage: \fig[width]{filename}{caption}{label}
% Example: \fig{media/image1.png}{Your caption here}{fig:mylabel}
% Example: \fig[0.8\linewidth]{media/image1.png}{Your caption here}{fig:mylabel}
\NewDocumentCommand{\fig}{O{1.0\linewidth} m m m}{%
  \begin{figure}[H]
    \centering
    \includegraphics[width=#1]{#2}
    \caption{#3}
    \label{#4}
  \end{figure}%
  \noindent\ignorespaces% Remove horizontal space and ignore following spaces
}


% SINGLE SPACING AND SMALL FONT FOR VERBATIM
\BeforeBeginEnvironment{verbatim}{\singlespacing\scriptsize}
\AfterEndEnvironment{verbatim}{\doublespacing\normalsize}

\endinput