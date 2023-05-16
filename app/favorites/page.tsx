import React from "react";
import EmptyState from "../components/EmptyState";
import getCurrentUser from "../actions/getCurrentUser";
import getFavoriteListings from "../actions/getFavoritesListing";
import FavoritesClient from "./FavoritesClient";

const ListingPage = async () => {
  const listing = await getFavoriteListings();
  const currentUser = await getCurrentUser();

  if (listing.length === 0) {
    return (
      <EmptyState
        title="No favorites"
        subtitle="Looks like you don't have favorites"
      />
    );
  }

  return (
    <div>
      <FavoritesClient listings={listing} currentUser={currentUser} />
    </div>
  );
};

export default ListingPage;
