# Tokensisation des joueurs

Chaque joueur de poker adopte un style de jeu unique. Afin d'améliorer la prise de décision du modèle, il serait pertinent de développer un tokeniseur 
capable de catégoriser les joueurs en fonction de leur style de jeu. Ce token servirait d'entrée au modèle, lui permettant d'affiner ses décisions en 
tenant compte des tendances spécifiques de chaque adversaire. Le transformer dédié à l'historique des actions des joueurs prendrai ce vecteur en entrée. 

Cela prendrait en entrée l'historique d'une main et en sortirait un token qui représenterait le style de jeu du joueur. Ce token serait ensuite mis à jour à 
chaque nouvelle main. 